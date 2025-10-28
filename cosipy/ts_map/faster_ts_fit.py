from pathlib import Path
import time

import numpy as np

from numba import jit, prange, get_num_threads, set_num_threads

import healpy as hp

from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame
import astropy.units as u
from scoords import Attitude


class FasterTSMap():
    """
    Likelihood mapper for GRBs

    Parameters
    ----------
    response : histpy.Histogram
        The instrument response

    """

    # global instance of solver object
    _fnf = None

    def __init__(self, response):

        self.CDS_axes = response.axes["Em", "Phi", "PsiChi"]

        self._rsp_nside = response.axes["NuLambda"].nside

        # make sure response is ordered as expected for Ei reduction.
        # Only do this if we have to, as it is expensive and makes a
        # copy of the response.
        axes_order = ["NuLambda", "Ei", "Em", "Phi", "PsiChi"]
        if list(response.axes.labels) != axes_order:
            print("WARNING: reordering response axes")
            response = response.project(axes_order)

        # flatten the CDS axes of the response and save the raw
        # values, which is all we need for analysis
        self._response = np.reshape(response.contents.value,
                                    response.shape[:2] + (-1,))

        self._rsp_sum = np.sum(self._response, axis=2)

        # initialize optimizer
        from .faster_norm_fit import FasterNormFit as fnf
        FasterTSMap._fnf = fnf()


    @staticmethod
    def get_hypothesis_coords(nside, pixels=None):

        """Get an array of hypothesis coordinates for a likelihood map
        in the galactic frame. The size of the map is defined by the
        Healpix nside parameter.

        Parameters
        ----------
        nside : int
            The nside of the map.

        Returns
        -------
        hypothesis_coords : array
            Array of hypothesis coordinates at the center of each pixel.

        """

        if pixels is None:
            npix = hp.nside2npix(nside)
            pixels = np.arange(npix, dtype=int)

        x, y, z = hp.pix2vec(nside, pixels, nest=True)

        return np.column_stack((x, y, z))


    @staticmethod
    def get_exposed_bins(src_loc, rsp_nside, orientations):
        """
        Get an array listing all Healpix map pixels in the *spaceraft
        frame* that are exposed to a GRB at galactic location src_loc,
        along with a weight representing the amount of exposure (time
        + interpolated location) that each receives.

        Inputs
        ------
           src_loc: np.ndarray of size 3
             Cartesian coordinates of the source location (a unit vector)
           rsp_nside: int
             The size of the Healpix grid on which we calculate exposure.
             Must match grid of instrument response.
           orientations: tuple of ndarrays
             Information on the spacecraft's orientations in the
             galactic frame during the time of the GRB.  A tuple
             (time, rot, max_angle, earth_zenith), where
               - time is an array of N times in seconds,
               - rot is an N x 3 x 3 array of rotation matrices that
                 transform galactic coords into spacecraft coords at each
                 time.
               - earth_zenith is an 3 x N array of N 3-vectors pointing to
                 the earth's zenith at each time
               - min_angle_cos is an array of N values giving cosine of
                 the angle w/the earth's zenith above which a source would
                 be blocked by the earth at each time

        Returns
        -------
           Tuple (pixels, weights)
           pixels: array of pixel indices in the Healpix grid with
                   nonzero exposure
           weights: total exposure (in seconds) for each pixel in pixels

        """

        dt, rot, earth_zenith, min_angle_cos = orientations

        # keep only time bins at which source is visible
        is_visible = (src_loc @ earth_zenith > min_angle_cos)
        rot = rot[is_visible]
        dt = dt[is_visible]

        # rotate source location into spacecraft frame
        # at each time step
        path_cart = rot @ src_loc

        # convert source path from Cartesian to spherical
        # (assumes r is always == 1)
        path_x = path_cart[:,0]
        path_y = path_cart[:,1]
        path_z = path_cart[:,2]

        path_theta = np.arccos(path_z)
        path_phi = np.arctan2(path_y, path_x)

        # interpolate path of source locations over time onto
        # Healpix grid
        pixels, weights = hp.get_interp_weights(rsp_nside,
                                                path_theta,
                                                path_phi)

        # compute total time spent within each pixel
        weights *= dt # weights apply to duration at each pixel

        unique_pixels, weights = \
            FasterTSMap.sparse_sum_duplicates(pixels.ravel(),
                                              weights.ravel(),
                                              dtype=np.float32)

        return unique_pixels, weights


    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True)
    def get_psr(response, rsp_sum, pixels, pix_weights, CDS_bins):
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
          response: 2D ndarray of size (n response pixels) x (CDS space size)
            The image response
          pix_weights: 1D ndarray of size (CDS space size)
            The exposure-time weight of each pixel
          CDS_bins: CDS bins to keep from each slice of response
        Returns
        -------
          - 1D ndarray of size |CDS_bins| with computed point response
          - sum of *all* CDS bins (whether in sparse set CDS_bins or not)
            in PSR

        """

        psr = np.zeros(len(CDS_bins), dtype = response.dtype)
        psr_sum = response.dtype.type(0)

        for i, w in zip(pixels, pix_weights):
            psr_sum += rsp_sum[i] * w
            rsp = response[i]
            for j in range(len(CDS_bins)):
                psr[j] += rsp[j] * w

        return psr, psr_sum


    @staticmethod
    def fast_ts_fit(src_loc):
        """Perform a TS fit on a single location at `src_loc`.

        Note that all parameters except src_loc are passed by setting
        class-scope variables (see setup_args() method below).  However,
        we document these additional parameters here.

        Parameters
        ----------
        src_loc: np.ndarray (size 3)
            Cartesian coordinates of proposed source location (a unit vector).
        response: 2D ndarray of dim (source map size) x (CDS space size)
            Instrument response, giving the expected intensities in
            each CDS bin for a burst of unit intensity at a given
            source location. (Prior computation has already averaged
            response over Ei using estimated integrated flux.)
        rsp_nside: int
            Number of sides for Healpix map corresponding to response
            CDS space.
        CDS_bins: np.ndarray of int
            Array of bins in CDS space with nonzero data weight. data
            and bkg_model weights are given only for these bins.
        data : np.ndarray (number of live CDS bins)
            The flattened, sparse Compton data space (CDS) array of the data,
            enumerating only weights of bins in CDS_bins.
        bkg_model : np.ndarray (number of live CDS bins)
            The flattened, sparse Compton data space (CDS) array of
            the background model
        orientations:
            The orientations of the spacecraft when data are collected.

        Returns
        -------
        Tuple holding TS fit results:
          (ts value, norm, failed)

        """

        response     = FasterTSMap._response
        rsp_sum      = FasterTSMap._rsp_sum
        rsp_nside    = FasterTSMap._rsp_nside
        CDS_bins     = FasterTSMap._CDS_bins
        data         = FasterTSMap._data
        bkg_model    = FasterTSMap._bkg_model
        orientations = FasterTSMap._orientations

        pixels, weights = FasterTSMap.get_exposed_bins(src_loc,
                                                       rsp_nside,
                                                       orientations)

        if len(pixels) > 0:
            psr, psr_sum = FasterTSMap.get_psr(response, rsp_sum,
                                               pixels, weights,
                                               CDS_bins)

            # fit the intensity of the GRB
            return FasterTSMap._fnf.solve(data, bkg_model, psr, psr_sum)
        else:
            # source is completely occluded
            return (0., 0., False)

    @staticmethod
    def get_orientation_history(ori_full, tstart, tend):
        """Read the history of detector orientations during a specified
        time window, and turn it into a convenient form for computing
        dwell maps.

        The returned orientation data consists of several arrays: an array
        of durations dt for each time bin between tstart and tend; an array
        with the spacecraft's orientation during that time bin, expressed as
        an inverse rotation matrix that maps galactic coordinates to
        instrument coordinates; an array of critical angles for earth
        occultation, and an array of SkyCoords pointing to earth's zenith

        Parameters
        ----------
        ori_full : np.ndarray
           array containing full orientation history of spacecraft
        tstart : float
           start time of interval to extract
        tend : float
           end time of interval to extract

        Returns
        -------
        orientation info: tuple of
          dt : array -- length of each orientation time bin
          rots : array -- inverse rotation matrices mapping map galactic to
           instrument coordinates for each time bin
          earth_zenith : array -- vector pointing to earth's zenith
           in each time bin
          min_angle_cos : array -- *cosine* of angle between source
            and earth zenith above which a source would be occulted by
            the earth, in each time bin; visible angles have cosines
            *larger* than this value

        """

        # get the orientation data and extract the interval corresponding
        # to the GRB
        times = ori_full[:,0]

        # find entries os and oe whose times bracket the interval;
        # add one to oe so we can use it as end of indexing range.
        # Range is guaranteed to have at least two successive times.

        os = np.searchsorted(times, tstart, side='left') - 1
        oe = np.searchsorted(times, tend, side='right') + 1

        event_times = times[os:oe]

        lat_x = ori_full[os:oe,1]
        lon_x = ori_full[os:oe,2]
        x_pointings = SkyCoord(l = lon_x, b = lat_x, unit = u.deg,
                               frame = "galactic")

        lat_z = ori_full[os:oe,3]
        lon_z = ori_full[os:oe,4]
        z_pointings = SkyCoord(l = lon_z, b = lat_z, unit = u.deg,
                               frame = "galactic")

        altitude = ori_full[os:oe, 5]

        lat_e = ori_full[os:oe,6]
        lon_e = ori_full[os:oe,7]
        e_pointings = SkyCoord(l = lon_e, b = lat_e, unit = u.deg,
                               frame = "galactic")

        dt, x_pointings, z_pointings, altitude, e_pointings = \
            FasterTSMap.interp_orientation(event_times,
                                           x_pointings, z_pointings,
                                           altitude, e_pointings,
                                           tstart, tend)

        attitude = Attitude.from_axes(x = x_pointings, z = z_pointings,
                                      frame = "galactic")
        rots = attitude.rot.inv().as_matrix()

        # Get max visible angle based on altitude
        r_earth = 6378.0 # earth radius in km
        max_angle = np.pi - np.arcsin(r_earth/(r_earth + altitude))

        return (dt, rots, e_pointings.cartesian.xyz.value, np.cos(max_angle))


    @staticmethod
    def interp_orientation(times,
                           x_pointings, z_pointings, altitude, e_pointings,
                           ts, te):
        """
        Interpolate an array of spacecraft orienations to a set of
        time bins.  We are given a list of times, the orientation at
        each time (including directions of the x and z axes, the
        altitude, and the earth zenith), and start and end times ts
        and te, which are assumed to satisfy

        times[0]  <  ts <= times[1]
        times[-2] <= te <  times[-1]

        We compute a set of time bins between successive input times
        after first trimming the first and last bins to start and end
        at ts and te, respectively.  For each time bin, we compute the
        x and z, and earth zenith directions and the altitude at the
        center of the bin by linear interpolation.

        Parameters
        ----------
        times : array of float
           Array of times at which orientation is reported
        x_pointings : array of SkyCoord
           Direction of the x axis (in galactic frame) at each time
        z_pointings : array of SkyCoord
           Direction of the z axis (in galactic frame) at each time
        altitiude: array of float
           Altitude of spacecraft at each time
        e_pointings : array of SkyCoord
           Direction of earth zenith (in galactic frame) at each time
        ts : float
           Start time for first bin
        te : float
           End time for last bin

        Returns
        -------
        tuple of time bin descriptors
        dt : array of float
           Length of each time bin
        x_pointings : array of SkyCoord
           Direction of x axis in middle of each time bin
        z_pointings : array of SkyCoord
           Direction of z axis in middle of each time bin
        altitiude: array of float
           Altitude of spacecraft in middle of each time bin
        e_pointings : array of SkyCoord
           Direction of earth zenith in middle of each time bin

        """

        def interp_pointings(pointings, times, ts, te):
            """
            Given a array of pointings (SkyCoord) corresponding to
            array of times, interpolate the pointings to the center of
            each time bin.  Replace the first and last bin edges with
            ts and te and use the corresponding interpolated pointings
            at these edges.
            """

            def vec_interp(v1, v2, alpha):
                v = v1 + alpha * (v2 - v1)
                return v / np.linalg.norm(v, axis=0)

            v = pointings.cartesian.xyz.value

            alpha_s = (ts - times[0])/(times[1] - times[0])
            vs = vec_interp(v[:,0], v[:,1], alpha_s)

            alpha_e = (te - times[-2])/(times[-1] - times[-2])
            ve = vec_interp(v[:,-2], v[:,-1], alpha_e)

            v[:,0] = vs
            v[:,-1] = ve

            v = vec_interp(v[:,:-1], v[:,1:], 0.5)
            pointings = SkyCoord(*v,
                                 representation_type='cartesian',
                                 frame='galactic')

            return pointings

        def interp_scalar(s, times, ts, te):

            def s_interp(s1, s2, alpha):
                return s1 + alpha * (s2 - s1)

            si = s.copy()

            alpha_s = (ts - times[0])/(times[1] - times[0])
            s_s = s_interp(s[0], s[1], alpha_s)

            alpha_e = (te - times[-2])/(times[-1] - times[-2])
            s_e = s_interp(s[-2], s[-1], alpha_e)

            si[0]  = s_s
            si[-1] = s_e

            return s_interp(si[:-1], si[1:], 0.5)

        altitude = interp_scalar(altitude, times, ts, te)
        x_pointings = interp_pointings(x_pointings, times, ts, te)
        z_pointings = interp_pointings(z_pointings, times, ts, te)
        e_pointings = interp_pointings(e_pointings, times, ts, te)

        times[0] = ts
        times[-1] = te

        return (np.diff(times),
                x_pointings, z_pointings,
                altitude, e_pointings)


    @staticmethod
    def sparse_sum_duplicates(indices, weights=None, dtype=None):
        """
        Given an array of indices, possibly with duplicates, and an
        optional array of weights per index (defaults to all ones if
        None), return a sorted array of the unique values in indices
        and, for each, the sum of the weights for each unique index.

        Parameters
        ----------
        indices : array of int
        weights : array of int or float type
        dtype : data type (optional)
           Type of returned weights.  If None, type is int if weights
           not given, or float64 if they are.

        Returns
        -------
          - unique_indices : array of int
             sorted unique indices in input
          - idx_weights : array of type as described above
             sum of weights for each unique index in input

        """

        if weights is None:
            unique_indices, idx_weights = np.unique(indices,
                                                    return_counts=True)
        else:
            sp_weights = np.bincount(indices, weights)
            unique_indices = np.flatnonzero(sp_weights)
            idx_weights = sp_weights[unique_indices]

        if dtype is not None:
            idx_weights = idx_weights.astype(dtype, copy=False)

        return unique_indices, idx_weights


    @staticmethod
    def trim_event_ends(events, bkg_rate):
        """
        Guess the detectable start and end time of a GRB represented
        in events, given an estimate of the local background rate, and
        discard events that occur before and after this end time.

        We assume that the background is Poisson with the specified
        rate.  We divide events into 1-second bins.

         * The burst is trimmed to start with the first 1-second bin
           whose count is significantly greater than expected for a
           Poisson rv with the given bkg_rate at the p=0.1 level.

         * The burst is trimmed to end before the first 1-second bin
           AFTER THE MAX OBSERVED VALUE whose count is *not*
           significantly greater than expected by the above criterion.

        For short bursts (< 2 seconds), we further trim their end
        based on the above criterion, but on a 0.1-second time scale.

        FIXME: the latter criterion assumes we look forward
        arbitrarily in time (to find the max) and should be replaced
        by a decision based only on prior history.

        Parameters
        ----------
        events : dict
          event dictionary (will be trimmed in place)
        bkg_rate : float
          estimated rate of bkg events per second

        """

        from scipy.stats import poisson

        event_times = events["time"].value

        t_start = event_times[0]
        t_end   = event_times[-1]

        # divide the events into 1-second bins
        t_edges = t_start + np.arange(t_end - t_start + 1, step=1)
        time_bins, _ = np.histogram(event_times, t_edges)

        # compute one minus the Poisson p-value for each bin
        bpois = poisson.cdf(time_bins, mu=bkg_rate)

        b_start = np.argmax(bpois >= 0.9) # returns 0 if none
        if bpois[b_start] < 0.9:
            print("Warning: no significant time bins found")

        b_max = np.argmax(time_bins)
        rest = time_bins[b_max:]
        bprest = bpois[b_max:]

        b_end = np.argmax(bprest < 0.9)
        if bprest[b_end] >= 0.9: # no low value
            b_end = len(rest)

        t_end    = t_start + b_max + b_end
        t_start += b_start

        #import os
        #t_end = np.minimum(t_end, t_start + int(os.environ["T_END"]))

        # further trim very short events using 0.1-second bins
        if t_end - t_start < 2:
            t_edges = t_start + np.arange(t_end - t_start + 0.1, step=0.1)
            time_bins, _ = np.histogram(event_times, t_edges)

            # compute one minus the Poisson p-value for each bin
            bpois = poisson.cdf(time_bins, mu=bkg_rate * 0.1)

            b_max = np.argmax(time_bins)
            rest = time_bins[b_max:]
            bprest = bpois[b_max:]

            b_end = np.argmax(bprest < 0.9)
            if bprest[b_end] >= 0.9: # no low value
                b_end = len(rest)

            t_end = t_start + (b_max + b_end) * 0.1

        e_start = np.searchsorted(event_times, t_start, side='left')
        e_end   = np.searchsorted(event_times, t_end,  side='right')

        events["time"]   = events["time"][e_start:e_end]
        events["Em"]     = events["Em"][e_start:e_end]
        events["Phi"]    = events["Phi"][e_start:e_end]
        events["PsiChi"] = events["PsiChi"][:,e_start:e_end]

        events["t_endpoints"] = (t_start, t_end)

        """
        # print some useful stats about the burst
        blen = t_end - t_start
        ev_rate = len(events['time'])/blen
        excess_rate = ev_rate - bkg_rate
        eev_count = np.round(excess_rate * blen)
        bev_count = len(events['time']) - eev_count

        #print(f"DURATION: {blen:.1f} s")
        #print(f"EXCESS EVENT RATE: {excess_rate:.1f} events/s")
        print(f"EXCESS {bev_count:.0f} {eev_count:.0f}")
        """

    @staticmethod
    def bin_events(events, axes, bkg_rate, mask=None, dtype=np.float64):
        """
        Bin a set of events, loaded as a dictionary by
        read_unbinned_data(), into a sparse array of flattened CDS
        bins.

        Parameters
        ----------
        events : dict
           unbinned event dictionary
        axes : Axes
           axes describing CDS dimensions on which to bin
        bkg_rate : float
           estimated rate of background events/sec during burst
        mask : np.array (optional)
           mask array; only keep CDS bins for which mask is nonzero
        dtype : numpy dtype (optional)
             type for binned event weights; default double precision
             float. We choose a floating-point type because we may
             in the future interpolate rather than just bin on the
             axes, at which point we'll need arbitrary real weights.

        Returns
        -------
        tuple (bins, weights)
           bins : array of int
             CDS bins with nonzero weight
           weights : array of type matching dtype
             values in each nonzero CDS bin

        """

        FasterTSMap.trim_event_ends(events, bkg_rate)

        # bin data on each CDS axis
        event_bins = (
            axes["Em"].find_bin(events["Em"]),
            axes["Phi"].find_bin(events["Phi"]),
            axes["PsiChi"].find_bin(theta = events["PsiChi"][0].value,
                                    phi = events["PsiChi"][1].value,
                                    lonlat = True)
        )

        # compute equivalent linear indices
        flat_bins = np.ravel_multi_index(event_bins, axes.shape)

        if mask is not None:
            # remove masked bins
            flat_bins = flat_bins[mask[flat_bins] != 0]

        # add weights (currently just # of occurrences) for each bin
        unique_bins, weights = FasterTSMap.sparse_sum_duplicates(flat_bins,
                                                                 dtype=dtype)

        return (unique_bins, weights)


    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
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
                       dtype=np.float32)

        for i in prange(response.shape[0]):
            for j in range(response.shape[1]):
                w = weights[j]
                for k, p in enumerate(bins):
                    rsp[i,k] += response[i,j,p] * w

        return rsp

    def setup_args(self, events, bkg_model, spectral_flux, orientations):
        """
        Set up the arguments for a TS mapping run.

        Event data is discretized to a sparse set of CDS bins, so that
        fitting can look at only these bins.

        We convolve the spectral flux with the response to average
        each PSR over the Ei dimension, resulting in a response matrix
        that is NuLambda x CDS.

        """

        # flatten the background model
        bkg_model = bkg_model.contents.ravel().astype(np.float32, copy=False)

        # compute sparse binning of events
        # We mask bins for which bkg_model is zero, because we cannot compute
        # a Poisson likelihood for an event whose rate is zero.  In future,
        # we may want to add pseudocounts to bkg_model to prevent this issue.
        CDS_bins, data = self.bin_events(events, self.CDS_axes,
                                         np.sum(bkg_model),
                                         bkg_model, dtype=np.float32)

        # Keep only those CDS bins of background model present in data
        bkg_model = bkg_model[CDS_bins]

        FasterTSMap._CDS_bins = CDS_bins
        FasterTSMap._data     = data

        ts, te = events["t_endpoints"]

        FasterTSMap._bkg_model    = bkg_model * (te - ts)
        FasterTSMap._orientations = self.get_orientation_history(orientations,
                                                                ts, te)

        FasterTSMap._rsp_nside    = self._rsp_nside

        # @ convolves flux with the *second* dimension of self._response
        #rsp = np.take(self._response, CDS_bins, axis=-1)
        #FasterTSMap._response = spectral_flux @ rsp
        FasterTSMap._response = self.selmul(self._response, CDS_bins, spectral_flux)

        FasterTSMap._rsp_sum = self._rsp_sum @ spectral_flux


    def parallel_ts_fit(self,
                        events,
                        bkg_model,
                        spectral_flux,
                        orientations,
                        nside = 16,
                        cpu_cores = None):

        """
        Evaluate LLR test statistic for a map at resolution nsides.

        Parameters
        ----------
        events : dict
           Observed events, in the form of a dictionary as read by
           read_unbinned_data()
        bkg_model : histpy.Histogram
           Local background model at the time of the burst
        spectral_flux : np.ndarray of size (# Ei bins)
           Integrated spectral flux of the source in each Ei bin
        orientations :
           History of spacecraft orientation in time bins during burst
        nside : resolution of final map
        cpu_cores : int > 0, optional
           The number of cpu cores you wish to use for the parallel
           computation (the default is `None`, which implies using all
           available cores to perform the computation).

        Returns
        -------
          ndarray of ts values for each pixel, in *NESTED* order

        """

        if cpu_cores is None:
            cpu_cores = get_num_threads()

        set_num_threads(cpu_cores) # limit Numba parallelism

        self.setup_args(events, bkg_model, spectral_flux, orientations)

        src_locs = FasterTSMap.get_hypothesis_coords(nside, None)

        results = [ FasterTSMap.fast_ts_fit(s) for s in src_locs ]

        ts = np.stack(results)[:,0]

        return ts


    def multires_ts_fit(self,
                        events,
                        bkg_model,
                        spectral_flux,
                        orientations,
                        nside_start = 1,
                        nside_max = 16,
                        containment = 0.90,
                        cpu_cores = None):
        """
        Evaluate LLR test statistic using multiresolution approach.
        We begin by evaluating the LLR at every pixel for nside =
        nside_start, then refine and rescore sufficiently high-scoring
        pixels (as determined by the containment threshold) and their
        neighbors at the next highest nside. Repeat until nside_max is
        reached.

        Parameters
        ----------
        events : dict
           Observed events, in the form of a dictionary as read by
           read_unbinned_data()
        bkg_model : histpy.Histogram
           Local background model at the time of the burst
        spectral_flux : np.ndarray of size (# Ei bins)
           Integrated spectral flux of the source in each Ei bin
        orientations :
           History of spacecraft orientation in time bins during burst
        nside_start : int (power of 2)
           initial resolution for result
        nside_max : int (power of 2)
           highest resolution for best-scoring parts of result
        containment: float
           containment fraction that determines which pixels to keep
           each time we increase nside. Only pixels scoring within
           the specified containment fraction of the best score found,
           along with their immediate neighbors, will be refined in the
           next round.
        cpu_cores : int > 0, optional
           The number of cpu cores you wish to use for the parallel
           computation (the default is `None`, which implies using all
           available cores to perform the computation).

        Returns
        -------
          Tuple (pixels, ts) where
            - pixels is an ndarray of UNIQ pixel IDs for a multiresolution map
            - ts is an ndarray of the ts values for each pixel

        """

        def nest2uniq(nside, pix):
            """
            Convert pixels from NEST to UNIQ indices
            given their nside.
            """

            return 4 * nside**2 + pix

        def refine(pix):
            """
            Given pixels in NEST format, expand each to
            its four sub-pixels at the next nside.

            """

            res = np.tile(pix, 4)
            res *= 4

            n = len(pix)
            res[n:2*n]   += 1
            res[2*n:3*n] += 2
            res[3*n:]    += 3

            return res

        if cpu_cores is None:
            cpu_cores = get_num_threads()

        set_num_threads(cpu_cores) # limit Numba parallelism

        self.setup_args(events, bkg_model, spectral_flux, orientations)

        cv_chi = self.get_chi_critical_value(containment=containment)

        pixels = np.arange(hp.nside2npix(nside_start), dtype=int)

        all_pix = []
        all_ts = []

        total_pix = 0
        nside = nside_start
        while nside <= nside_max:

            src_locs = FasterTSMap.get_hypothesis_coords(nside, pixels)

            total_pix += len(src_locs)

            results = [ FasterTSMap.fast_ts_fit(s) for s in src_locs ]

            ts = np.stack(results)[:,0]

            if nside == nside_max:
                # Done -- save all remaining pixels and their values
                all_pix.append(nest2uniq(nside, pixels))
                all_ts.append(ts)
                break

            # keep pixels exceeding a significance threshold
            hi_mask = (ts >= ts.max() - cv_chi)

            # keep any pixel that is adjacent to one that exceeds
            # the significance threshold
            hi_adj = hp.get_all_neighbours(nside, pixels[hi_mask], nest=True)
            hi_adj = np.unique(hi_adj)
            adj_mask = np.isin(pixels, hi_adj, assume_unique=True)
            hi_mask[adj_mask] = True

            # For pixels that we will not refine,
            # compute their unique indices and save them.
            lo_mask = ~hi_mask
            lo_pix = nest2uniq(nside, pixels[lo_mask])
            lo_ts  = ts[lo_mask]

            all_pix.append(lo_pix)
            all_ts.append(lo_ts)

            # Refine kept pixels to corresponding ranges of
            # pixels at next nside
            pixels = refine(pixels[hi_mask])

            nside *= 2

        # print(total_pix)
        return (np.concatenate(all_pix), np.concatenate(all_ts))


    @staticmethod
    def get_chi_critical_value(containment = 0.90):

        """
        Get the critical value of the chi^2 distribution based ob the
        confidence level.

        Parameters
        ----------
        containment : float, optional
            The confidence level of the chi^2 distribution (the
            default is `0.9`, which implies that the 90% containment
            region).

        Returns
        -------
        float
            The critical value corresponds to the confidence level.

        """
        from scipy.stats import chi2

        return chi2.ppf(containment, df=2)


    @staticmethod
    def plot_ts(m_ts, true_src_loc = None,
                containment = None, dpi = 300,
                save_plot = False, save_dir = "",
                save_name = "ts_map.png"):

        """
        Plot a TS map.

        Parameters
        ----------
        m_ts: numpy.ndarray
          The array of ts values from parallel ts fit
        true_src_loc : astropy.coordinates.SkyCoord, optional
           The true location of the source (the default is `None`,
           which implies that there are no coordinates to be printed
           on the TS map).
        containment : float, optional
           The containment level of the source (the default is `None`,
           which will plot raw TS values).
        dpi : int, optional
           The dpi for plotting.
        save_plot : bool, optional
           Set `True` to save the plot (default `False`)
        save_dir : str or pathlib.Path, optional
           The directory to save the plot.
        save_name : str, optional
           The file name of the plot to be save.

        """

        import matplotlib.pyplot as plt

        # get plotting canvas
        fig, ax = plt.subplots(dpi=dpi)
        max_ts = np.max(m_ts)

        # plot the ts map with containment region
        if containment is not None:
            critical = FasterTSMap.get_chi_critical_value(containment)

            #pixmask = m_ts >= max_ts - critical
            #for i, v in enumerate(m_ts):
            #    if pixmask[i]:
            #        p = np.exp(v - max_ts)
            #        print(f"{i} {p}")

            hp.mollview(m_ts, nest = True,
                        max = max_ts, min = max_ts-critical,
                        title = f"Containment {100*containment}%",
                        coord = "G", hold = True)
        else:
            hp.mollview(m_ts, nest = True,
                        max = max_ts, min = np.min(m_ts[m_ts > 0]),
                        coord = "G", hold = True)

        if true_src_loc is not None:
            # mark GRB location in galactic coords
            lon = true_src_loc.l.deg
            lat = true_src_loc.b.deg

            hp.projscatter(lon, lat, marker = "x", linewidths = 0.5,
                           lonlat=True, coord = "G",
                           label = f"True location at l={lon}, b={lat}",
                           color = "fuchsia")

        # mark l=0, b=0 in galactic coordinates
        #hp.projscatter(0, 0, marker = "o", linewidths = 0.5,
        #               lonlat=True, coord = "G", color = "red")
        #hp.projtext(350, 0, "(l=0, b=0)",
        #            lonlat=True, coord = "G", color = "red")

        if save_plot == True:
            fig.savefig(Path(save_dir) / save_name, dpi = dpi)


    @staticmethod
    def plot_multi_ts(m_ts, m_uniq, true_src_loc = None,
                      containment = None, dpi = 300,
                      save_plot = False, save_dir = "",
                      save_name = "ts_map.png"):

        """
        Plot a multiresolution TS map.

        Parameters
        ----------
        m_ts: numpy.ndarray (float)
          The array of ts values from multiresolution ts fit
        m_uniq: nump.ndarray (int)
          The UNIQ pixel ides for the map from multiresolution ts fit
        true_src_loc : astropy.coordinates.SkyCoord, optional
           The true location of the source (the default is `None`,
           which implies that there are no coordinates to be printed
           on the TS map).
        containment : float, optional
           The containment level of the source (the default is `None`,
           which will plot raw TS values).
        dpi : int, optional
           The dpi for plotting.
        save_plot : bool, optional
           Set `True` to save the plot (default `False`)
        save_dir : str or pathlib.Path, optional
           The directory to save the plot.
        save_name : str, optional
           The file name of the plot to be save.

        """

        import matplotlib.pyplot as plt
        import mhealpy as mhp

        # get plotting canvas
        fig = plt.figure(dpi=dpi)
        axMoll = fig.add_subplot(1,1,1, projection="mollview")

        ts_map = mhp.HealpixMap(data=m_ts, uniq=m_uniq)

        max_ts = np.max(m_ts)

        # plot the ts map, with containment region if specified
        if containment is not None:
            axMoll.set_title(f"Containment {100*containment}%")

            critical = FasterTSMap.get_chi_critical_value(containment)
            min_ts = max_ts - critical

        else:
            axMoll.set_title("Mollweide view")

            min_ts = np.min(m_ts[m_ts > 0])

        ts_map.plot(ax=axMoll, vmax = max_ts, vmin = min_ts)

        # force colorbar ticks to same format as hp.mollview
        cb = axMoll.images[-1].colorbar
        from matplotlib import ticker
        cb.formatter = ticker.FormatStrFormatter("%g")
        cb.ax.set_xticks([min_ts, max_ts])

        if true_src_loc is not None:
            # mark GRB location in galactic coords
            lon = true_src_loc.l.deg
            lat = true_src_loc.b.deg
            axMoll.scatter(lon, lat, marker = "x", linewidths = 0.5,
                           label = f"True location at l={lon}, b={lat}",
                           color = "fuchsia",
                           transform = axMoll.get_transform('world'))

        # mark zenith in galactic coords
        axMoll.scatter(0, 0, marker="o", linewidths=0.5,
                       color = "red",
                       transform = axMoll.get_transform('world'))
        axMoll.text(350, 0,  "(l=0, b=0)",
                    color = "red",
                    transform = axMoll.get_transform('world'))

        if save_plot == True:
            fig.savefig(Path(save_dir) / save_name, dpi = dpi)
