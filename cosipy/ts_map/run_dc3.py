from pathlib import Path
import sys

import numpy as np

import time

from histpy import Histogram

import mhealpy as hp

from cosipy.ts_map.read_dc3_data import (
    read_burst_params,
    read_unbinned_events,
    combine_unbinned_events,
    get_local_bkg_model,
    read_orientation_history
)

from cosipy.ts_map import FasterTSMap

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


np.random.seed(0)

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

orientation_path = data_dir / "orientation.npy"

response_path = data_dir / "response_hist.h5"

num_cpus = 8

def angular_error(pmax, true_src_loc):

    def uniq2nest(u):
        # stolen from astropy's healpix module
        level = int( np.log2(u//4) // 2 )
        ipix = u - (1 << 2*(level + 1))
        return (1 << level, ipix)

    pmax_nside, pmax_nest = uniq2nest(pmax)

    pmax_center = hp.pix2vec(ipix=pmax_nest, nside=pmax_nside, nest=True)
    true_loc    = true_src_loc.cartesian.xyz.value

    best_src_dist = np.rad2deg(np.arccos(np.dot(pmax_center, true_loc)))

    return best_src_dist

print("READING RESPONSE")
response = Histogram.open(response_path)

ts = FasterTSMap(response = response)

# Run a few warmup iterations to make sure the JIT runs and avoid
# other startup transients.  Empirically, 5 is the minimum number of
# iterations needed on ARM to avoid seeing an artificially long
# running time for the first recorded iteration.
n_warmup = 5

results = []
for i, burst in enumerate([bursts[-1]] * n_warmup + bursts):

    AMP = float(sys.argv[1])

    # retrieve ground truth for the burst
    params_path = grb_dir / burst / (burst + "_params.txt")

    spectrum, true_src_loc = read_burst_params(params_path)

    # retrieve the unbinned burst events
    signal_path = grb_dir / burst / (burst + "_signal.hdf5")
    background_path = grb_dir / burst / (burst + "_bkg.hdf5")

    signal_events = read_unbinned_events(signal_path, max_events = AMP)
    bkg_events = read_unbinned_events(background_path)

    # fetch a random set of background events sufficient to amplify
    # the existing background by a factor of AMP.  Set their times
    # to random values between real bkg start and end and add them to
    # the real background.
    #prior_background_path = grb_dir / burst / (burst + "_bkg_prior.hdf5")
    #prior_bkg_events = read_unbinned_events(prior_background_path,
    #                                        max_events = int(len(bkg_events["time"]) * (AMP - 1)))
    #btime_s, btime_e = np.min(bkg_events["time"]), np.max(bkg_events["time"])
    #prior_bkg_events["time"] = btime_s + np.random.rand(len(prior_bkg_events["time"])) * (btime_s - btime_e)
    #bkg_events = combine_unbinned_events(bkg_events, prior_bkg_events)

    events = combine_unbinned_events(signal_events, bkg_events)

    #print("T_ACTUAL:",
    #      (signal_events["time"][-1] - signal_events["time"][0]).value)

    # get the local background model

    prior_background_path = grb_dir / burst / (burst + "_bkg_prior.hdf5")

    bkg_model = get_local_bkg_model(bkg_model_path,
                                    prior_background_path)

    # HACK: adjust intensity of bkg model to match observed bkg events
    #bkg_rate = len(bkg_events["time"]) / (btime_e - btime_s)
    #bkg_model *= bkg_rate / np.sum(bkg_model)

    # get the orientation history of the detector
    orientations = read_orientation_history(orientation_path)

    t_start = time.time()

    # compute (rough!) Ei spectral flux approximation as a histogram of the
    # Em values in the observed events
    spectral_flux, _ = np.histogram(events["Em"],
                                    bins = response.axes["Em"].edges)
    spectral_flux = spectral_flux.astype(np.float32)

    ts_mr_pix, ts_mr_llrs = ts.multires_ts_fit(events,
                                               bkg_model,
                                               spectral_flux,
                                               orientations,
                                               nside_start = 4,
                                               nside_max = 64,
                                               containment = 0.999,
                                               cpu_cores = num_cpus)
    t_end = time.time()

    t = t_end - t_start

    imax = np.argmax(ts_mr_llrs)
    pmax = ts_mr_pix[imax]

    err = angular_error(pmax, true_src_loc)

    if i >= n_warmup:
        ts.plot_multi_ts(ts_mr_llrs, ts_mr_pix,
                         true_src_loc = true_src_loc,
                         containment = 0.999, dpi = 300,
                         save_plot = True, save_dir = "maps",
                         save_name = "ts_map_" + burst + ".png")

        save_moc_map(ts_mr_llrs, ts_mr_pix,
                     out_nside = 64,
                     save_dir = "maps",
                     save_name = "ts_map_" + burst + ".txt")

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
