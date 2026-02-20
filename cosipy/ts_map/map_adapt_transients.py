from pathlib import Path
import time
import sys

import numpy as np
from scipy.stats import chi2

from astropy.table import Table

import mhealpy as hp

import h5py as h5

from histpy import Histogram

from cosipy import FastTSMap, MOCTSMap
from cosipy.response import GalacticResponse
from cosipy.response.functions import get_integrated_spectral_model

from cosipy.ts_map.read_dc3_data import (
    read_burst_params,
    read_unbinned_events,
    combine_unbinned_events,
)

from sphdist import moc_expected_angdist

def save_moc_map(llrs, uniq_pix, out_nside, save_name,
                 save_dir, live_threshold=1e-3):

    max_llr = np.max(llrs)
    min_diff = chi2.ppf(0.999, df=2)
    min_prob = np.exp(-min_diff)
    CUTOFF = min_prob

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

    live = (probs >= CUTOFF)
    probs = probs[live]
    pix = pix[live]

    lons, lats = hp.pix2ang(out_nside, pix, nest=True, lonlat=True)

    with h5.File(save_dir / (save_name + ".h5"), "w") as f:
        f.attrs["source"] = np.array([true_src_loc.b.deg, true_src_loc.l.deg])

        f.create_dataset("pixel",       data=pix,
                         dtype=int, compression="gzip")
        f.create_dataset("latitude",    data=lats,
                         dtype=np.float32, compression="gzip")
        f.create_dataset("longitude",   data=lons,
                         dtype=np.float32, compression="gzip")
        f.create_dataset("probability", data=probs,
                         dtype=np.float32, compression="gzip")

    return np.sum(probs >= live_threshold)


data_dir = Path("/project/cassini/adapt_grbs")

transient_path = data_dir / "adapt_transients"

output_dir = Path("/project/cassini/adapt_grbs")

output_path = output_dir / "adapt_maps"
output_path.mkdir(parents=True, exist_ok=True)

model_dir = Path("/project/cassini/adapt_grbs")

bkg_model_path = model_dir / "adapt_bkg_model.h5"

response_path = model_dir / "adapt_response_w_area-nn.h5"

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


###############################################################

np.random.seed(1957)

print("Opening detector response...", file=sys.stderr)
response = GalacticResponse.open(response_path, dtype=np.float32)

# create mapping object
mapper = MOCTSMap(response,
                  cds_frame="galactic",
                  response_in_memory = True)
map_nside = 64

moc_strategy = \
    MOCTSMap.PaddingStrategy(
        MOCTSMap.ContainmentStrategy(0.999)
    )

sources = list(transient_path.glob("adapt_*_source.h5"))

# Run a few warmup iterations to make sure the JIT runs and avoid
# other startup transients.  Empirically, 5 is the minimum number of
# iterations needed on ARM to avoid seeing an artificially long
# running time for the first recorded iteration.
n_warmup = 3
sources = [sources[0]]*n_warmup + sources

print("alt,az,transient_id,n_src,n_bkg,err,exp_angdist,conf,time", flush=True)

results = []
for i, signal_file in enumerate(sources):

    # extract source info from name
    fields = signal_file.name.split("_")
    p = int(fields[1][1:])
    a = int(fields[2][1:])
    inst = fields[3]
    prefix = f"adapt_p{p}_a{a}_{inst}"

    params_path = transient_path / f"adapt_p{p}_a{a}_params.txt"
    spectrum, true_src_loc = read_burst_params(params_path)

    ts = 1580000000. # arbitrary value used by generator script
    te = ts + 1.  # CHEAT: we use the real burst length, not an estimate

    # retrieve the unbinned burst events
    signal_path = signal_file
    background_path = transient_path / f"{prefix}_background.h5"

    signal_events = read_unbinned_events(signal_path)
    n_src = len(signal_events["time"])
    bkg_events = read_unbinned_events(background_path)
    n_bkg = len(bkg_events["time"])
    events = combine_unbinned_events(signal_events, bkg_events)

    # CHEAT: we use the actual instead of the prior estimated background
    bkg_rate = len(bkg_events["time"]) / (te - ts)

    bkg_model = Histogram.open(bkg_model_path)
    bkg_model = bkg_model.project(("Em", "Phi", "PsiChi"))
    bkg_model *= bkg_rate

    # compute total expected bkg fluence during transient
    bkg_model *= te - ts

    # CHEAT: we use the real spectrum, not a guess
    spectral_flux = get_integrated_spectral_model(spectrum,
                                                  response.axes["Ei"])

    t_start = time.time()

    #m_llrs = mapper.fit_unbinned(ts, te, events, bkg_model,
    #                             spectral_flux,
    #                             nside = map_nside,
    #                             cpu_cores = num_cpus)
    #m_pix = hp.nest2uniq(nside=map_nside,
    #                     ipix=np.arange(len(m_llrs), dtype=int))

    m_llrs, m_pix = mapper.fit_unbinned(ts, te, events, bkg_model,
                                        spectral_flux,
                                        max_nside = map_nside,
                                        strategy = moc_strategy,
                                        cpu_cores = num_cpus)
    t_end = time.time()

    t = t_end - t_start

    if i >= n_warmup:

        mapper.plot_ts(m_llrs, m_pix,
                       skycoord = true_src_loc,
                       grid_lines = False,
                       plot_zenith = False,
                       dpi = 300,
                       save_plot = True,
                       save_dir = output_path,
                       save_name = f"{prefix}_map.png")

        n_live_pix = save_moc_map(m_llrs, m_pix,
                                  out_nside = 64,
                                  save_dir = output_path,
                                  save_name = f"{prefix}_map")

        imax = np.argmax(m_llrs)
        pmax = m_pix[imax]

        b = hp.HealpixBase(uniq=m_pix, scheme="NUNIQ", coordsys="G")
        p_true = b.ang2pix(theta=np.pi/2 - true_src_loc.b.rad,
                           phi=true_src_loc.l.rad)
        p_llr = m_llrs[p_true]

        err = moc_angular_error(pmax, true_src_loc)

        max_llr = np.max(m_llrs)
        conf =  chi2.cdf(max_llr - p_llr, df=2)

        m_probs = np.exp(m_llrs - max_llr)
        m_probs = m_probs / np.sum(m_probs) # normalize to sum to 1

        exp_angdist = moc_expected_angdist(np.pi/2 - true_src_loc.b.rad, # src colatitude
                                           true_src_loc.l.rad,           # src longitude
                                           m_pix, m_probs)
        exp_angdist = np.rad2deg(exp_angdist)

        print(f"{90-p},{a},{inst},{n_src},{n_bkg},{err:.3f},{exp_angdist:.3f},{conf:.4f},{t:.3f}", flush=True)
