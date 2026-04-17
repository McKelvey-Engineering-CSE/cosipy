from pathlib import Path

from enum import Enum

import numpy as np
import numba

import matplotlib.pyplot as plt

import healpy as hp
from mhealpy import HealpixBase

from astropy.time import Time

from cosipy import SpacecraftHistory

from .fast_norm_fit import FastNormFit as fnf

import logging
logger = logging.getLogger(__name__)


class FastTSMap():

    class Frame(Enum):
        LOCAL = 1
        GALACTIC = 2

    def __init__(self, response,
                 orientation = None,
                 cds_frame = "local",
                 energy_channel = None,
                 max_cache_size = None,
                 response_in_memory = False):

        """
        Initialize the instance of a TS map fit.

        Parameters
        ----------
        response : FullDetectorResponse or GalacticResponse
            Detector response
        orientation : cosipy.SpacecraftHistory, optional
            Orientation history of spacecraft; required for "local"
            cds_frame, not used if frame is "galactic"
        cds_frame : str, optional
            frame of directions used for PsiChi axis of CDS.  One of
            "local" (frame attached to spacecraft) or "galactic".
            Default is local.
        energy_channel : 2-element list, of form
                         [lower_channel, upper_channel], optional
            Energy (Em) channels to use in fitting (Python range
            lower_channel:upper_channel). If not specified, use all
            Em channels.
        max_cache_size : int, optional
            Maximum number of entries to store in PSRCache; if None,
            no limit

        """

        match cds_frame:
            case "local":
                self._cds_frame = FastTSMap.Frame.LOCAL
                if orientation is None:
                    raise TypeError("When data are binned in local frame, "
                                    "orientation must be provided")
                orientation.cache_earth_occ = True
                self._orientation = orientation

            case "galactic":
                self._cds_frame = FastTSMap.Frame.GALACTIC
                self._orientation = None

            case _:
                raise TypeError(f"Unrecognized frame {cds_frame}, "
                                "must be 'local' or 'galactic'")

        if energy_channel is None:
            self._em_slice = slice(None)
        else:
            self._em_slice = slice(energy_channel[0], energy_channel[1])

        self._response = response

        # set correct callback for computing PSRs based on
        # whether we're caching the full response in memory
        self.response_in_memory = response_in_memory
        if response_in_memory:
            self.get_psr = self.get_psr_in_mem

            # preload the full response
            self._psrs, self._psr_sums = \
                self._preload_response(response, self._em_slice)
        else:
            self.get_psr = self.get_psr_out_mem

        labels = self._response.axes.labels

        # mapping only works with CDS's consisting of Em/Phi/PsiChi
        # (in any order). The response must map from NuLambda / Ei to
        # the CDS.

        if not all(labels[:2] == ("NuLambda", "Ei")):
            raise ValueError("Response axes must begin with (NuLambda, Ei)")

        # extract order of response's CDS dimensions for linearization
        # of data, bkg

        self.cds_order = tuple(labels[2:])
        if not all(ax in ("Em", "Phi", "PsiChi") for ax in self.cds_order):
            raise ValueError("Response CDS axes must be Em/Phi/PsiChi")

        self._fnf = fnf(max_iter=1000)

        self._max_cache_size = max_cache_size

    @staticmethod
    def _get_hypothesis_coords(nside, pixels = None,
                               scheme = "nested",
                               coordsys = "galactic"):
        """
        Get directions corresponding to pixels of a HEALPix map of a
        given resolution and scheme.

        Parameters
        ----------
        nside : int
            Nside of HEALPix map
        pixels : array-like of int, optional
            Array of pixels to convert to directions; if not
            specified, directions will be generated for every pixel in
            map
        scheme : str, optional
            Scheme of HEALPix map ("ring" or "nested"; default: nested)
        coordsys : str, optional
            Coordinate system of HEALPix map (default: galactic)

        Returns
        -------
        hypothesis_coords : np.ndarray of (# pixels x 3)
            Cartesian 3-vectors for each pixel's direction

        """

        if pixels is None:
            npix = hp.nside2npix(nside)
            pixels = np.arange(npix, dtype=int)

        hpbase = HealpixBase(nside = nside, scheme = scheme,
                             coordsys = coordsys)

        return np.column_stack(hpbase.pix2vec(pixels))

    @staticmethod
    def _get_cds_array(hist, em_slice):
        """
        Convert a CDS histogram to a flattened array, keeping
        just the selected channels of the Em dimension.

        Parameters
        -----------
        hist : histpy.Histogram
           A CDS count Histogram
        em_slice : Slice object
           Energy (Em) channels to use in fitting

        Returns
        -------
        cds_array : numpy.ndarray
            Flattened CDS array

        """

        hist_cds_sliced = hist.slice[{"Em" : em_slice}]
        hist_cds = hist_cds_sliced

        cds_array = hist_cds.contents
        if hist_cds.unit is not None:
            cds_array = cds_array.value

        return cds_array.ravel()

    def _fit_one_direction(self, source, orientation,
                           data_cds_array, bkg_model_cds_array,
                           psr_cache):
        """
        Perform a TS fit of data for a single source direction

        Parameters
        ----------
        source : np.ndarray
            source direction as Cartesian 3-vector
        data_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the data.
        bkg_model_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the
            background model.
        psr_cache : PSRCache
            Cache to retrieve PSR for source direction

        Returns
        -------
        result of TS fitting:
          [ts value, norm, norm_err, failed, # iterations]

        """

        if self._cds_frame == FastTSMap.Frame.LOCAL:
            # get list of HEALPix pixels with nonzero exposure from source
            pixels, exposures = \
                orientation.get_exposure(source = source,
                                         base = self._response,
                                         earth_occ = True,
                                         dtype = self._response.dtype)
            exposures = exposures.value

        else: # galactic frame

            # convert source vector to polar coords
            x, y, z = source
            lon   = np.arctan2(y, x)
            colat = np.arccos(z)

            # interpolate the source onto the response grid
            pixels, exposures = \
                self._response.get_interp_weights(theta = colat,
                                                  phi = lon,
                                                  lonlat = False)
            exposures = exposures.astype(self._response.dtype, copy=False)

        if len(pixels) > 0:
            ei_cds_array, ei_sum = \
                self.get_psr(psr_cache, pixels, exposures)

            if ei_sum > 0: # some pixels may have no data in response
                return self._fnf.solve(data_cds_array, bkg_model_cds_array,
                                       ei_cds_array, ei_sum)

        # default: return nothing
        return (0., 0., 0., False)

    @classmethod
    def get_psr_in_mem(cls, psr_cache, pixels, exposures):
        rsp, rsp_sum = psr_cache
        return cls._get_psr_in_mem(rsp, rsp_sum, pixels, exposures)

    @staticmethod
    def get_psr_out_mem(psr_cache, pixels, exposures):
        # sum the PSRs for each NuLambda pixel according to their
        # exposure weights
        ei_cds_array = np.zeros(psr_cache.shape, psr_cache.dtype)
        ei_sum = psr_cache.dtype.type(0.)

        for p, exposure in zip(pixels, exposures):
            if p < psr_cache.npixels: # allow for response w/o all NuLambda pixels
                psr, psr_sum = psr_cache.get_psr(p)
                ei_cds_array += psr * exposure
                ei_sum += psr_sum * exposure

        return ei_cds_array, ei_sum

    def _prepare_inputs(self, data, bkg_model, spectral_flux):
        """
        Prepare the data and background arrays for ts fitting, and get ready
        to read and cache PSRs for different source directions.  The shape
        and contents of the arrays and PSRs depends on the data reductions

        implied by the energy channel and spectrum.

        Parameters
        ----------
        data : histpy.Histogram
            Observed data, which includes counts from both signal and
            background.
        bkg_model : histpy.Histogram
            Model used to estimate background counts in observed data.
        spectral_flux : np.ndarray of float
            Integrated spectral flux of source in response's Ei bins

        Returns
        -------
        data_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the data.
        bkg_model_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the
            background model.
        psr_cache : PSRCache
            Cache to retrieve PSR for source directions

        """

        # make sure data and background CDS are ordered to match response
        data = data.todense().project(self.cds_order).astype(self._response.dtype, copy=False)
        bkg_model = bkg_model.todense().project(self.cds_order).astype(self._response.dtype, copy=False)

        # get the flattened data and background CDS arrays
        data_cds_array = self._get_cds_array(data, self._em_slice)
        bkg_model_cds_array = self._get_cds_array(bkg_model, self._em_slice)

        # eliminate CDS cells with no counts in data (due to data
        # sparsity) or in bkg model (lack of pseudocounts in bkg model
        # -- could be considered a bug, may cause divide-by-zero error
        # in fitting)
        valid_cells = np.where(np.logical_and(data_cds_array > 0,
                                              bkg_model_cds_array > 0))[0]

        data_cds_array = data_cds_array[valid_cells]
        bkg_model_cds_array = bkg_model_cds_array[valid_cells]

        if self.response_in_memory:
            psr_cache = self.reduce_response(self._psrs, self._psr_sums,
                                             valid_cells, spectral_flux)
        else:
            psr_cache = PSRCache(self._response,
                                 self._em_slice,
                                 valid_cells,
                                 spectral_flux,
                                 max_size = self._max_cache_size)

        return data_cds_array, bkg_model_cds_array, psr_cache


    def _prepare_inputs_unbinned(self, events, bkg_model, spectral_flux):
        """
        Prepare the observed events and background model for ts fitting,
        and get ready to read and cache PSRs for different source
        directions.  The shape and contents of the arrays and PSRs
        derived from the input depends on the data reductions implied
        by the energy channel and spectrum.

        Parameters
        ----------
        events: dict
            Observed events for transient and background combined.
        bkg_model : histpy.Histogram
            Model used to estimate background counts in observed data.
        spectral_flux : np.ndarray of float
            Integrated spectral flux of source in response's Ei bins

        Returns
        -------
        data_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the data.
        bkg_model_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the
            background model.
        psr_cache : PSRCache
            Cache to retrieve PSR for source directions

        """

        # get the flattened background CDS array
        bkg_model = bkg_model.todense().project(self.cds_order).astype(self._response.dtype, copy=False)
        bkg_model_cds_array = self._get_cds_array(bkg_model, self._em_slice)

        cds_axes = self._response.axes[self.cds_order]

        # bin data on each CDS axis
        event_bins = []
        for label in self.cds_order:
            if label == "PsiChi":
                # PsiChi binning is done on HEALPix axis
                bins = cds_axes[label].find_bin(theta = events["PsiChi"][0].value,
                                                phi = events["PsiChi"][1].value,
                                                lonlat = True)
            else:
                bins = cds_axes[label].find_bin(events[label])

            event_bins.append(bins)

        if self._em_slice != slice(None):
            # apply energy channel filter to events
            e_lo, e_hi = self._em_slice.start, self._em_slice.stop
            e_lo = 0 if e_lo is None else e_lo
            e_hi = em_axis.nbins if e_hi is None else e_hi
            em_axis = cds_axes.label_to_index("Em")
            ev_mask = ((event_bins[em_axis] >= e_lo) &
                       (event_bins[em_axis] < e_hi))
            event_bins = [b[ev_mask] for b in event_bins]

        # linearize array of CDS bins and suppress any for which bkg
        # model has zero weight (since these would otherwise produce
        # an infinite Poisson likelihood)
        flat_event_bins = np.ravel_multi_index(event_bins, cds_axes.shape)
        flat_event_bins = flat_event_bins[bkg_model_cds_array[flat_event_bins] != 0]

        # add weights (currently just # of occurrences) for each bin
        valid_cells, data_cds_array = \
            SpacecraftHistory._sparse_sum_duplicates(flat_event_bins,
                                                     dtype=self._response.dtype)

        # keep only bkg model bins for which data event count is
        # nonzero
        bkg_model_cds_array = bkg_model_cds_array[valid_cells]

        if self.response_in_memory:
            psr_cache = self.reduce_response(self._psrs, self._psr_sums,
                                             valid_cells, spectral_flux)
        else:
            psr_cache = PSRCache(self._response,
                                 self._em_slice,
                                 valid_cells,
                                 spectral_flux,
                                 max_size = self._max_cache_size)

        return data_cds_array, bkg_model_cds_array, psr_cache

    def fit(self, data, bkg_model, spectral_flux,
            nside = 16, cpu_cores = None):
        """
        Produce a ts map of specified resolution.

        Parameters
        ----------
        data : histpy.Histogram
            Observed data, which includes counts from both signal and
            background.
        bkg_model : histpy.Histogram
            Model used to estimate background counts in observed data.
            Should give ABSOLUTE expected counts over duration of transient
            for each CDS bin.
        spectral_flux : np.ndarray of float
            Integrated spectral flux of source in response's Ei bins
        nside : int, optional
            HEALPix nside of ts map to produce (default 16)
        cpu_cores : int, optional
            Number of processors to use (default: do not restrict)

        Returns
        -------
        results : numpy.ndarray
            Fitted ts values for each hypothesis coordinate

        """

        if cpu_cores is not None:
            numba.set_num_threads(min(cpu_cores,
                                      numba.config.NUMBA_NUM_THREADS))

        data_cds_array, bkg_model_cds_array, psr_cache = \
            self._prepare_inputs(data, bkg_model, spectral_flux)

        if self._cds_frame == FastTSMap.Frame.LOCAL:
            # compute possible source dirs in same frame
            # we will use to translate them to local-frame paths
            hyp_frame = self._orientation.attitude.frame
        else: # galactic frame
            hyp_frame = "galactic"

        hypothesis_coords = self._get_hypothesis_coords(nside,
                                                        coordsys=hyp_frame)

        results = [
            self._fit_one_direction(source,
                                    self._orientation,
                                    data_cds_array,
                                    bkg_model_cds_array,
                                    psr_cache)[0]
            for source in hypothesis_coords
        ]

        return np.array(results)

    def fit_unbinned(self, ts, te, events, bkg_model, spectral_flux,
                     nside = 16, cpu_cores = None):
        """
        Produce a ts map of specified resolution from unbinned events.

        Parameters
        ----------
        ts : float
            start time of transient (UNIX secs)
        te : float
            end time of transient (UNIX secs)
        events : dict
            Observed events for transient and background combined.
        bkg_model : histpy.Histogram
            Model used to estimate background counts in observed data.
            Should give ABSOLUTE expected counts over duration of transient
            for each CDS bin.
        spectral_flux : np.ndarray of float
            Integrated spectral flux of source in response's Ei bins
        nside : int, optional
            HEALPix nside of ts map to produce (default 16)
        cpu_cores : int, optional
            Number of processors to use (default: do not restrict)

        Returns
        -------
        results : numpy.ndarray
            Fitted ts values for each hypothesis coordinate

        """

        if cpu_cores is not None:
            numba.set_num_threads(min(cpu_cores,
                                      numba.config.NUMBA_NUM_THREADS))

        if self._cds_frame == FastTSMap.Frame.LOCAL:
            orientation = self._orientation.source_interval(Time(ts, format="unix"),
                                                            Time(te, format="unix"))
            # compute possible source dirs in same frame
            # we will use to translate them to local-frame paths
            hyp_frame = orientation.attitude.frame
        else:
            # galactic frame
            orientation = None
            hyp_frame = "galactic"

        data_cds_array, bkg_model_cds_array, psr_cache = \
            self._prepare_inputs_unbinned(events, bkg_model, spectral_flux)

        hypothesis_coords = self._get_hypothesis_coords(nside,
                                                        coordsys=hyp_frame)

        results = [
            self._fit_one_direction(source,
                                    orientation,
                                    data_cds_array,
                                    bkg_model_cds_array,
                                    psr_cache)[0]
            for source in hypothesis_coords
        ]

        return np.array(results)


    @staticmethod
    def plot_ts(m_ts, skycoord = None, containment = None, scheme="nested",
                plot_zenith = True, save_plot = False, save_dir = "",
                save_name = "ts_map.png", dpi = 300):
        """
        Plot a TS map.

        Parameters
        ----------
        m_ts : numpy.ndarray
            The array of ts values from a ts fit.
        skycoord : astropy.coordinates.SkyCoord, optional
            The true location of the source (default: do not plot)
        containment : float, optional
            Restrict the plotted pixels to the specified containment
            threshold relative to the max ts value (default: plot
            *all* ts values)
        scheme : string, optional
            HEALPix scheme of ts map values ("ring" or "nested";
            default = "nested")
        plot_zenith: bool, optional
            If true, plot and label zenith
        save_plot : bool, optional
            Save the plot to a file (default: False)
        save_dir : string, optional
            Directory in which to save the plot
        save_name : str, optional
            File name under which tos ave the plot
        dpi : int, optional
            DPI used for plotting / saving

        """

        fig, ax = plt.subplots(dpi = dpi)
        nest = scheme.startswith("nest")

        if containment is not None:
            critical = FastTSMap.get_chi_critical_value(containment = containment)
            max_ts = np.max(m_ts)
            hp.mollview(m_ts, max = max_ts, min = max_ts - critical,
                        nest=nest,
                        title = f"Containment {containment*100}%",
                        coord = "G",
                        hold = True)
        else:
            hp.mollview(m_ts, nest=nest, coord = "G", hold = True)

        if skycoord is not None:
            lon = skycoord.l.deg
            lat = skycoord.b.deg
            hp.projscatter(lon, lat, marker = "x",
                           linewidths = 0.5,
                           lonlat=True,
                           coord = "G",
                           label = f"True location at l={lon}, b={lat}",
                           color = "fuchsia")

        if plot_zenith:
            hp.projscatter(0, 0, marker = "o",
                           linewidths = 0.5,
                           lonlat=True,
                           coord = "G",
                           color = "red")

            hp.projtext(350, 0, "(l=0, b=0)",
                        lonlat=True,
                        coord = "G",
                        color = "red")

        if save_plot:
            fig.savefig(Path(save_dir)/save_name, dpi = dpi)

        plt.show()
        plt.close(fig)

    @staticmethod
    def get_chi_critical_value(containment = 0.90):
        """
        Get the critical value of the chi^2 distribution based on the
        confidence level.

        Parameters
        ----------
        containment : float, optional
          The confidence level of the chi^2 distribution (the default is
          `0.9`, which implies that the 90% containment region).

        Returns
        -------
        float
            The critical value corresponding to the confidence level.

        """

        from scipy.stats import chi2

        return chi2.ppf(containment, df=2)


    def _preload_response(self, response, em_slice):

        # read the response matrix into memory
        rsp = response.to_dr()

        # keep just requested slice on Em axis (if this becomes
        # important, we can hack to_dr() to only load the requested
        # slice from disk)
        if em_slice is not slice(None):
            rsp = rsp.slice[{"Em" : em_slice}]

        # reduce to bare array
        rsp = rsp.contents.value

        # linearize all but NuLambda and Ei dimensions
        rsp = rsp.reshape(rsp.shape[:2] + (-1,))

        return rsp, np.sum(rsp, axis=-1)

    def reduce_response(self, rsp, rsp_sum, valid_cells, spectral_flux):
        flux = spectral_flux.contents.value.astype(rsp.dtype, copy=False)
        rsp = self.selmul(rsp, valid_cells, flux)
        rsp_sum = rsp_sum @ flux

        return rsp, rsp_sum

    @staticmethod
    @numba.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def selmul(response, bins, weights):
        """
        Compute the equivalent of
            rsp = weights @ response[:,:,bins]
        That is, we compute
             rsp[i,k] = sum_j weights[j] * response[i, j, bins[k]]

        Doing this directly in Numba and parallelizing it across cores
        is faster than first doing the selection of bins, then using
        OpenBLAS to do the multiply.  It also avoids an intermediate
        copy of the response during selection.

        """

        rsp = np.zeros(shape=(response.shape[0], len(bins)),
                       dtype=response.dtype)

        for i in numba.prange(response.shape[0]):
            for j in range(response.shape[1]):
                w = weights[j]
                for k, p in enumerate(bins):
                    rsp[i,k] += response[i,j,p] * w

        return rsp

    @staticmethod
    @numba.jit(nopython=True, nogil=True, fastmath=True)
    def _get_psr_in_mem(rsp, rsp_sum, pixels, pix_weights):
        """
        Compute a point-source response as a sum of response slices for
        each instrument-frame pixel exposed to the source, werighted
        by the length of its exposure.

        We implement in Numba to avoid intermediate copies of response
        slices when weighting and summing to get the PSR.  Ideally,
        we'd use a dot product call, but we know of no Numpy idiom to
        perform dot while skipping the slicing/multiplication when the
        weight is zero, and anything else creates an intermediate copy
        of each slice when multiplying by w.

        Parameters
        ----------
          rsp: 2D ndarray of size (n response pixels) x (CDS space size)
            The image response
          pix_weights: 1D ndarray of size (n response pixels)
            The exposure-time weight of each pixel

        Returns
        -------
          - 1D ndarray of (CDS space size) with computed PSR
          - sum over *all* CDS bins (whether in sparse CDS space or
            not) in PSR

        """

        n_cds_bins = rsp.shape[-1]
        psr = np.zeros(n_cds_bins, dtype = rsp.dtype)
        psr_sum = rsp.dtype.type(0)

        for i, w in zip(pixels, pix_weights):
            if i < len(rsp_sum): # allow for response w/o all NuLambda pixels
                psr_sum += rsp_sum[i] * w
                rspi = rsp[i]
                for j in range(n_cds_bins):
                    psr[j] += rspi[j] * w

        return psr, psr_sum


class PSRCache:
    """
    A cached reader for PSR data from a response file, designed for
    use with FastTSMap.  For a given NuLambda pixel p, we fetch the
    pixel's data from the underlying response file and do all the data
    reduction needed to compute a PSR for pixel p averaged over the
    input flux.  The result is cached so that, when different source
    directions require a PSR for the same pixel p, we don't do the
    fetching and reduction more than once.

    If memory usage is a concern, the cache can be set to a given max
    size with LRU replacement.  But the reduced PSR sums away the Ei
    and Em dimensions *and* removes CDS voxels that do not matter for
    the ts_map computation, so it is much smaller than a raw chunk of
    the response file. Hence, it is likely not necessary to limit the
    cache size in practice.

    """

    def __init__(self, response, em_slice, valid_cells, flux,
                 max_size = None):
        """
        Create a new PSRCache, providing the information needed
        to fetch and reduce PSRs from the response file on demand.

        Parameters
        ----------
        response : FullDetectorResponse
          The response from which to read slices for PSR computation
        em_slice : Slice object
          The slice of the Em axis used to compute PSRs
        valid_cells : np.ndarray of int
          CDS voxels on the linearized Phi/PsiChi axis that are actually
          used in in the ts_map computation
        flux : Histogram
          Integrated spectral flux, binned according to response's Ei axis
        max_size: int (optional)
          If not None, maximum number of NuLambda pixels for which we will
          cache PSRs.  The cache is managed according to an LRU policy.

        """

        from collections import OrderedDict

        self.cache = OrderedDict()
        self.max_size = max_size

        self.response = response
        self._npixels = response.axes["NuLambda"].nbins
        self.em_axis = response.axes.label_to_index("Em") - 1 # for NuLambda
        self.em_slice = em_slice
        self.valid_cells = valid_cells

        self.ei_weights = \
            flux.contents.value.astype(self.dtype, copy=False) * \
            response.eff_area_correction

        #self.nLookups = 0
        #self.nMisses = 0

    @property
    def shape(self):
        """
        Array shape of a PSR returned by the cache
        """
        return (len(self.valid_cells),)

    @property
    def dtype(self):
        """
        Element type of a PSR returned by the cache
        """
        return self.response.dtype

    @property
    def npixels(self):
        """
        Number of valid NuLambda pixels in the response
        """
        return self._npixels

    def get_psr(self, p):
        """
        Get the reduced PSR for NuLambda pixel p.

        Parameters
        ----------
        p : int
          NuLambda value of requested PSR

        Returns
        -------
        psr : np.ndarray of float (length = |valid_cells|)
          PSR for pixel p, summed over requested Em and
          convolved with spectral flux.  The result gives
          one value per valid voxel.
        psr_sum
          Sum of PSR for pixel p over *all* voxels, not just
          the valid ones.

        """

        #self.nLookups += 1
        v = self.cache.get(p)
        if v is None: # cache miss
            #self.nMisses += 1
            v = self._compute_psr(p)
            self.cache[p] = v

            # implement LRU policy if requested
            if self.max_size is not None and len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        else:
            # move MRU value to end to support LRU policy if requested
            if self.max_size is not None:
                self.cache.move_to_end(p)

        return v

    '''
    def print_stats(self):
        """
        Print cache miss statistics
        """

        missRate = 0. if self.nLookups == 0 else self.nMisses/self.nLookups

        print(f"Cache size: {len(self.cache)} (out of {self.max_size})")
        print(f"Misses: {self.nMisses} / {self.nLookups} = {missRate:0.3f}")
    '''

    def _compute_psr(self, p):
        """
        Compute a reduced PSR for NuLambda pixel p.

        Returns
        -------
        psr : np.ndarray of float (length = |valid_cells|)
          PSR for pixel p, summed over requested Em and
          convolved with spectral flux.  The result gives
          one value per valid voxel.
        psr_sum
          Sum of PSR for pixel p over *all* voxels, not just
          the valid ones.

        """

        # get raw CDS counts for pixel, trimmed by Em slice size
        # is Ei x CDS dims
        counts = self.response.get_counts(p, self.em_slice)

        # convert to float : Ei x CDS dims
        counts = counts.astype(self.response.dtype, copy=False)

        # linearize CDS : Ei x CDS voxels. Note that we ensure in
        # FastTSMap that data and bkg will use the same dimension
        # ordering as the response for the CDS, so there is no need to
        # re-order dimensions here.
        counts = counts.reshape(counts.shape[0], -1)

        # extract valid CDS voxels of psr after capturing sum of *all*
        # voxels : Ei x valid CDS voxels
        psr_sum = np.sum(counts, axis=1)
        psr = counts[:, self.valid_cells]

        # convolve psr with flux (and also eff_area correction
        # weights, which have not yet been applied) to remove Ei
        # dimension
        psr_sum = np.dot(psr_sum, self.ei_weights)
        psr = np.tensordot(psr, self.ei_weights, axes=(0,0))

        return psr, psr_sum
