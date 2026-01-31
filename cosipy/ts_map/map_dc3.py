from pathlib import Path
import time

import numpy as np

import mhealpy as hp

from cosipy import SpacecraftFile, FastTSMap, MOCTSMap
from cosipy.response import FullDetectorResponse
from cosipy.response.functions import get_integrated_spectral_model

from cosipy.ts_map.read_dc3_data import (
    read_burst_params,
    read_unbinned_events,
    combine_unbinned_events,
    get_local_bkg_model,
)

bursts = [
    "bn080802386",
    "bn081207680",
    "bn090424592",
    "bn100612726",
    "bn110605183",
    "bn131122490",
    "bn140329295",
    "bn161004964",
    "bn170405777",
    "bn180504136",
    "bn180703876",
    "MGF051103",
    "MGF070201",
    "MGF070222",
    "MGF180128A",
    "MGF200415A",
    "MGF231115A",
]

def save_moc_map(llrs, uniq_pix, out_nside, save_name, save_dir = ""):

    CUTOFF = 1e-8

    max_llr = np.max(llrs)
    probs = np.exp(llrs - max_llr)
    probs = probs / np.sum(probs) # normalize to sum to 1

    # Expand each multiresolution pixel to the maximum
    # nside.  Divide the probability equally among all
    # higher-level pixels expanded from a lower-level one.
    los, his = hp.uniq2range(out_nside, uniq_pix)
    allpix = []
    allprobs = []
    for lo, hi, p in zip(los, his, probs):
        npix = hi - lo
        pnew = p / npix
        allpix.append(np.arange(lo, hi, dtype=int))
        allprobs.append(np.full(npix, pnew))
    pix = np.concatenate(allpix)
    probs = np.concatenate(allprobs)

    #nsides, pix = hp.uniq2nest(uniq_pix)

    live = (probs >= CUTOFF)
    probs = probs[live]
    pix = pix[live]

    print("LIVE:", np.sum(probs >= 1e-3))

    lons, lats = hp.pix2ang(out_nside, pix, nest=True, lonlat=True)

    with open(save_dir + "/" + save_name, "w") as f:

        print(f"# nside {out_nside} NEST", file=f)
        print("# pix_id lat lon prob", file=f)
        for pix, lat, lon, prob in zip(pix, lats, lons, probs):
            print(f"{pix} {lat:.3f} {lon:.3f} {prob}", file=f)


data_dir = Path("/project/cassini/cosidata/dc3")
#data_dir = Path("/home/jbuhler/dc3")

grb_dir = data_dir / "grb"

bkg_model_path = data_dir / "bg" / "binned_bg.hdf5"

orientation_path = data_dir / "orientation.fits"

response_path = data_dir / "response.h5"

num_cpus = 8

def angular_error(nside, pmax, true_src_loc):

    pmax_center = hp.pix2vec(ipix=pmax, nside=nside, nest=True)
    true_loc    = true_src_loc.cartesian.xyz.value

    best_src_dist = np.rad2deg(np.arccos(np.dot(pmax_center, true_loc)))

    return best_src_dist

def moc_angular_error(pmax, true_src_loc):

    pmax_nside, pmax_nest = hp.uniq2nest(pmax)

    pmax_center = hp.pix2vec(ipix=pmax_nest, nside=pmax_nside, nest=True)
    true_loc    = true_src_loc.cartesian.xyz.value

    best_src_dist = np.rad2deg(np.arccos(np.dot(pmax_center, true_loc)))

    return best_src_dist

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

    return t_start, t_end

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

###############################################################

np.random.seed(0)

# get the orientation history of the detector
print("Reading orientation history...")
orientations = SpacecraftFile.open(orientation_path)

print("Opening detector response...")
response = FullDetectorResponse.open(response_path, dtype=np.float32)

# create mapping object
mapper = MOCTSMap(response, orientations)
map_nside = 64

moc_strategy = \
    MOCTSMap.PaddingStrategy(
        MOCTSMap.ContainmentStrategy(0.99)
    )

# Run a few warmup iterations to make sure the JIT runs and avoid
# other startup transients.  Empirically, 5 is the minimum number of
# iterations needed on ARM to avoid seeing an artificially long
# running time for the first recorded iteration.
n_warmup = 1

results = []
for i, burst in enumerate([bursts[-1]] * n_warmup + bursts):

    # retrieve ground truth for the burst
    params_path = grb_dir / burst / (burst + "_params.txt")

    spectrum, true_src_loc = read_burst_params(params_path)

    # retrieve the unbinned burst events
    signal_path = grb_dir / burst / (burst + "_signal.hdf5")
    background_path = grb_dir / burst / (burst + "_bkg.hdf5")
    signal_events = read_unbinned_events(signal_path)
    bkg_events = read_unbinned_events(background_path)
    events = combine_unbinned_events(signal_events, bkg_events)

    # get background
    prior_background_path = grb_dir / burst / (burst + "_bkg_prior.hdf5")

    # get_local_bkg_model returns expected RATE of bkg events / sec in
    # each CDS bin during burst
    bkg_model = get_local_bkg_model(bkg_model_path,
                                    prior_background_path)

    # adjust ends of transient based on signal above bg
    # modifies events in place; returns start and end times
    ts, te = trim_event_ends(events, np.sum(bkg_model))

    # compute total expected bkg fluence during transient
    bkg_model *= te - ts

    spectral_flux = get_integrated_spectral_model(spectrum,
                                                  response.axes["Ei"])
    t_start = time.time()

    # Note that FastTSMap expects the response *path* because we're not
    # reading the whole thing into memory now.  We could still alter
    # the fitting code to take the events and bkg_model instead of
    # passing them to init(), so that we only have to open the response
    # once.

    #llrs = mapper.fit_unbinned(ts, te, events, bkg_model, spectral_flux,
    #                           nside = map_nside, cpu_cores = num_cpus)

    m_llrs, m_pix = mapper.fit_unbinned(ts, te, events, bkg_model, spectral_flux,
                                        max_nside = map_nside,
                                        strategy = moc_strategy,
                                        cpu_cores = num_cpus)
    t_end = time.time()

    t = t_end - t_start

    if i >= n_warmup:
        imax = np.argmax(m_llrs)
        pmax = m_pix[imax]

        err = moc_angular_error(pmax, true_src_loc)

        mapper.plot_ts(m_llrs, m_pix,
                       skycoord = true_src_loc,
                       grid_lines = False,
                       plot_zenith = False,
                       dpi = 300,
                       save_plot = True, save_dir = "maps",
                       save_name = "ts_map_" + burst + ".png")

        #save_moc_map(m_llrs, m_pix,
        #             out_nside = 64,
        #             save_dir = "maps",
        #             save_name = "ts_map_" + burst + ".txt")

        results.append((burst, err, t))
        print(f"{burst} {err:.3f} {t:.3f}", flush=True)
    else:
        print(f"WARMUP {i+1}/{n_warmup}")


err_avg = 0.
t_avg = 0.
for r in results:

    _, err, t = r

    err_avg += err
    t_avg += t

n = len(bursts)
err_avg /= n
t_avg /= n

print(f"\nAVG {err_avg:.3f} {t_avg:.3f}")
