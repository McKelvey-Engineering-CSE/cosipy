from pathlib import Path

import numpy as np
import numba

import mhealpy as hp

import matplotlib.pyplot as plt

from astropy.time import Time

from .fast_ts_fit import FastTSMap, Frame

import logging
logger = logging.getLogger(__name__)

class MOCTSMap(FastTSMap):
    """
    Multi-resolution source mapping.
    """

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
        orientation : cosipy.SpacecraftFile, optional
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

        super().__init__(response,
                         orientation = orientation,
                         cds_frame = cds_frame,
                         energy_channel = energy_channel,
                         max_cache_size = max_cache_size,
                         response_in_memory = response_in_memory)


    class Strategy:
        """
        A generic strategy API for selecting pixels to refine in a moc map.
        """

        def __call__(self, ts, pixels, nside):
            """
            Select a subset of input pixels to refine.

            Parameters
            ----------
            ts : np.array of float
               ts values for a set of pixels
            pixels : np.array of int
               nested-scheme indices for pixels corresponding to each ts
               value
            nside : int
               nside of map from which pixels are drawn

            Returns
            -------
              boolean mask -- True only for those pixels that should be
              refined

            """
            raise RuntimeError("Strategy subclass must redefine select()")

    class TopKStrategy(Strategy):
        """
        Refine a fixed number of pixels with the highest ts values
        """

        def __init__(self, k):
            """
            Parameters
            ----------
            k : int
              refine the k pixels with highest ts values
            """

            self.k = k

        def __call__(self, ts, pixels, nside):
            # eliminate pixels with zero ts, which are not worth
            # expanding and can tickle implementation-specific
            # behavior around ties in the kth highest ts score
            k = np.minimum(self.k, np.count_nonzero(ts) - 1)

            hi_idx = np.argpartition(ts, -k)[-k:]
            hi_mask = np.zeros(len(ts), dtype=bool)
            hi_mask[hi_idx] = True

            return hi_mask

    class ContainmentStrategy(Strategy):
        """
        Refine all pixels within a specified containment region
        based on ts value
        """

        def __init__(self, containment):
            """
            Parameters
            ----------
            containment : float
              refine pixels whose ts value is within the specified
              containment region vs the max value.
            """

            self.chi = FastTSMap.get_chi_critical_value(containment)

        def __call__(self, ts, pixels, nside):
            return (ts >= ts.max() - self.chi)

    class PaddingStrategy(Strategy):
        """
        After applying a specified strategy, pad the result to include
        all neighbors of pixels chosen to be refined.
        """

        def __init__(self, sub_strategy):
            """
            Parameters
            ----------
            sub_strategy : MOCTSMap.Strategy subclass
              strategy to apply prior to padding
            """

            self.sub_strategy = sub_strategy

        def __call__(self, ts, pixels, nside):

            hi_mask = self.sub_strategy(ts, pixels, nside)

            hi_adj = hp.get_all_neighbours(nside, pixels[hi_mask], nest=True)
            hi_adj = np.unique(hi_adj)
            adj_mask = np.isin(pixels, hi_adj, assume_unique=True)
            hi_mask[adj_mask] = True

            return hi_mask

    def fit(self, data, bkg_model, spectral_flux,
            max_nside = 16, init_nside = 1, strategy = None,
            cpu_cores = None):
        """
        Construct a multi-resolution map of ts statistics, selectively
        refining the highest-scoring pixels.

        Parameters
        ----------
        data : histpy.Histogram
            Observed data, which includes counts from both signal and
            background.
        bkg_model : histpy.Histogram
            Model used to estimate background counts in observed data
        spectral_flux : np.ndarray of float
            Integrated spectral flux of source in response's Ei bins
        max_nside : int, optional
          highest possible nside reached during refinement (default 16)
        init_nside : int, optional
          lowest nside used in map (default 1)
        strategy : MOCTSMap.Strategy subclass, optional
          strategy to use in selecting pixels to refine. If None,
          default to TopKStrategy with k=8
        cpu_cores : int, optional
          number of processors to use (default: do not restrict)

        Returns
        -------
        ts : np.ndarray
          ts statistics for each pixel in map
        uniqs : np.ndarray of int
          uniq pixel IDs for each output pixel in map

        """

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

        if strategy is None:
            self.strategy = self.TopKStrategy(k=8)
        else:
            self.strategy = strategy

        if cpu_cores is not None:
            numba.set_num_threads(cpu_cores)

        data_cds_array, bkg_model_cds_array, psr_cache = \
            self._prepare_inputs(data, bkg_model, spectral_flux)

        all_pix = []
        all_ts = []

        # initially, compute ts for all pixels at minimum nside
        nside = init_nside
        pixels = np.arange(hp.nside2npix(init_nside), dtype=int)

        while nside <= max_nside:

            if self._cds_frame == Frame.LOCAL:
                # compute possible source dirs in same frame
                # we will use to translate them to local-frame paths
                hyp_frame = self._orientation.frame
            else: # galactic frame
                hyp_frame = "galactic"

            src_locs = self._get_hypothesis_coords(nside, pixels,
                                                   coordsys=hyp_frame)

            results = [
                self._fit_one_direction(source,
                                        self._orientation,
                                        data_cds_array,
                                        bkg_model_cds_array,
                                        psr_cache)[0]
                for source in src_locs
            ]

            ts = np.array(results)

            if nside == max_nside:
                # Done -- save all remaining pixels and their values
                all_pix.append(hp.nest2uniq(nside, pixels))
                all_ts.append(ts)
                break

            hi_mask = self.strategy(ts, pixels, nside)

            # For pixels that we will *not* refine, compute their
            # unique indices and save them.
            lo_mask = ~hi_mask
            lo_pix = hp.nest2uniq(nside, pixels[lo_mask])
            lo_ts  = ts[lo_mask]

            all_pix.append(lo_pix)
            all_ts.append(lo_ts)

            # Split pixels that we *will* refine down to next nside
            pixels = refine(pixels[hi_mask])

            nside *= 2

        return np.concatenate(all_ts), np.concatenate(all_pix)

    def fit_unbinned(self, ts, te, events, bkg_model, spectral_flux,
                     max_nside = 16, init_nside = 1, strategy = None,
                     cpu_cores = None):
        """
        Construct a multi-resolution map of ts statistics, selectively
        refining the highest-scoring pixels.

        Parameters
        ----------
        ts : float
            start time of transient (UNIX secs)
        te : float
            end time of transient (UNIX secs)
        events : dict
            Observed events for transient and background combined.
        bkg_model : histpy.Histogram
            Model used to estimate background counts in observed data
        spectral_flux : np.ndarray of float
            Integrated spectral flux of source in response's Ei bins
        max_nside : int, optional
          highest possible nside reached during refinement (default 16)
        init_nside : int, optional
          lowest nside used in map (default 1)
        strategy : MOCTSMap.Strategy subclass, optional
          strategy to use in selecting pixels to refine. If None,
          default to TopKStrategy with k=8
        cpu_cores : int, optional
          number of processors to use (default: do not restrict)

        Returns
        -------
        ts : np.ndarray
          ts statistics for each pixel in map
        uniqs : np.ndarray of int
          uniq pixel IDs for each output pixel in map

        """

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

        if strategy is None:
            self.strategy = self.TopKStrategy(k=8)
        else:
            self.strategy = strategy

        if cpu_cores is not None:
            numba.set_num_threads(cpu_cores)

        if self._cds_frame == FastTSMap.Frame.LOCAL:
            orientation = self._orientation.source_interval(Time(ts, format="unix"),
                                                            Time(te, format="unix"))
        else:
            orientation = None

        data_cds_array, bkg_model_cds_array, psr_cache = \
            self._prepare_inputs_unbinned(events, bkg_model, spectral_flux)

        all_pix = []
        all_ts = []

        # initially, compute ts for all pixels at minimum nside
        nside = init_nside
        pixels = np.arange(hp.nside2npix(init_nside), dtype=int)

        while nside <= max_nside:
            src_locs = self._get_hypothesis_coords(nside, pixels)

            results = [
                self._fit_one_direction(source,
                                        orientation,
                                        data_cds_array,
                                        bkg_model_cds_array,
                                        psr_cache)[0]
                for source in src_locs
            ]

            ts = np.array(results)

            if nside == max_nside:
                # Done -- save all remaining pixels and their values
                all_pix.append(hp.nest2uniq(nside, pixels))
                all_ts.append(ts)
                break

            hi_mask = self.strategy(ts, pixels, nside)

            # For pixels that we will *not* refine, compute their
            # unique indices and save them.
            lo_mask = ~hi_mask
            lo_pix = hp.nest2uniq(nside, pixels[lo_mask])
            lo_ts  = ts[lo_mask]

            all_pix.append(lo_pix)
            all_ts.append(lo_ts)

            # Split pixels that we *will* refine down to next nside
            pixels = refine(pixels[hi_mask])

            nside *= 2

        return np.concatenate(all_ts), np.concatenate(all_pix)

    @staticmethod
    def plot_ts(moc_ts, moc_uniq,
                skycoord = None, containment = None,
                grid_lines = True, plot_zenith = True,
                save_plot = False, save_dir = "",
                save_name = "ts_map.png", dpi = 300):
        """
        Plot a multi-resolution TS map.

        Parameters
        ----------
        moc_ts : np.ndarray
            ts values of multiresolution map
        moc_uniq : np.ndarray of int
            uniq HEALPixel values of multiresolution map
        skycoord : astropy.coordinates.SkyCoord, optional
            The true location of the source (default: do not plot)
        containment : float, optional
            Restrict the plotted pixels to the specified containment
            threshold relative to the max ts value (default: plot
            *all* ts values)
        grid_lines : bool, optional
            Print lines bordering each pixel in the map (default: True)
        plot_center: bool, optional
            If true, plot and label coords (0,0) on map (default: True)
        save_plot : bool, optional
            Save the plot to a file (default: False)
        save_dir : string, optional
            Directory in which to save the plot
        save_name : str, optional
            File name under which tos ave the plot
        dpi : int, optional
            DPI used for plotting / saving

        """

        moc_map = hp.HealpixMap(data = moc_ts, uniq = moc_uniq)

        # get plotting canvas
        fig = plt.figure(dpi=dpi)
        axMoll = fig.add_subplot(1,1,1, projection="mollview")

        max_ts = np.max(moc_ts)

        # plot the ts map, with containment region if specified
        if containment is not None:
            axMoll.set_title(f"Containment {100*containment}%")

            critical = FastTSMap.get_chi_critical_value(containment)
            min_ts = max_ts - critical

        else:
            axMoll.set_title("Mollweide view")

            min_ts = np.min(moc_ts[moc_ts > 0])

        moc_map.plot(ax=axMoll, vmax = max_ts, vmin = min_ts)

        if grid_lines:
            moc_map.plot_grid(ax = plt.gca(), color = 'grey',
                              linewidth = 0.1);

        # force colorbar ticks to same format as hp.mollview
        cb = axMoll.images[-1].colorbar
        from matplotlib import ticker
        cb.formatter = ticker.FormatStrFormatter("%g")
        cb.ax.set_xticks([min_ts, max_ts])

        if skycoord is not None:
            # mark GRB location in galactic coords
            lon = skycoord.l.deg
            lat = skycoord.b.deg
            axMoll.scatter(lon, lat, marker = "x", linewidths = 0.5,
                           label = f"True location at l={lon}, b={lat}",
                           color = "fuchsia",
                           transform = axMoll.get_transform('world'))

        if plot_zenith:
            # mark zenith in galactic coords
            axMoll.scatter(0, 0, marker="o", linewidths=0.5,
                           color = "red",
                           transform = axMoll.get_transform('world'))
            axMoll.text(350, 0,  "(l=0, b=0)",
                        color = "red",
                        transform = axMoll.get_transform('world'))

        if save_plot:
            fig.savefig(Path(save_dir) / save_name, dpi = dpi)

        plt.close()
