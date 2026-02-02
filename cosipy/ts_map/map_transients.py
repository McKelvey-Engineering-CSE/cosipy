from pathlib import Path
import time
import sys

import numpy as np

from astropy.table import Table

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

from sphdist import moc_expected_angdist

def save_moc_map(llrs, uniq_pix, out_nside, save_name,
                 save_dir, live_threshold=1e-3):

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

    live = (probs >= CUTOFF)
    probs = probs[live]
    pix = pix[live]

    lons, lats = hp.pix2ang(out_nside, pix, nest=True, lonlat=True)

    tbl = Table((pix, lats, lons, probs),
                names=("pixel","lat","lon","probability"))
    tbl.write(save_dir / (save_name + ".h5"), overwrite=True)

    return np.sum(probs >= live_threshold)

data_dir = Path("/home/jbuhler/dc3")

output_dir = Path("/home/jbuhler/ts_map_data")

output_path = output_dir / "maps"
output_path.mkdir(parents=True, exist_ok=True)

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


###############################################################

np.random.seed(1957)

# get the orientation history of the detector
print("Reading orientation history...", file=sys.stderr)
orientations = SpacecraftFile.open(orientation_path)

print("Opening detector response...", file=sys.stderr)
response = FullDetectorResponse.open(response_path, dtype=np.float32)

# create mapping object
mapper = MOCTSMap(response, orientations,
                  response_in_memory = True)
map_nside = 64

moc_strategy = \
    MOCTSMap.PaddingStrategy(
        MOCTSMap.ContainmentStrategy(0.99)
    )

transient_path = output_dir / "transients"
sources = list(transient_path.glob("sim_*_params.txt"))

# Run a few warmup iterations to make sure the JIT runs and avoid
# other startup transients.  Empirically, 5 is the minimum number of
# iterations needed on ARM to avoid seeing an artificially long
# running time for the first recorded iteration.
n_warmup = 3
sources = [sources[0]]*n_warmup + sources

print(f"n_events,transient_len,transient_id, n_live_pix,err,exp_angdist,time", flush=True)

results = []
for i, param_file in enumerate(sources):

    # extract source info from name
    fields = param_file.name.split("_")
    n_events = int(fields[1])
    transient_len = float(fields[2].replace("-","."))
    transient_id = int(fields[3])
    prefix = f"sim_{n_events}_{int(transient_len)}_{transient_id}"

    params_path = transient_path / f"{prefix}_params.txt"
    spectrum, true_src_loc, endpts = read_burst_params(params_path)
    ts, te = endpts

    # retrieve the unbinned burst events
    signal_path = transient_path / f"{prefix}_source.h5"
    background_path = transient_path / f"{prefix}_bkg.h5"
    signal_events = read_unbinned_events(signal_path)
    bkg_events = read_unbinned_events(background_path)
    events = combine_unbinned_events(signal_events, bkg_events)

    # get background
    prior_background_path = transient_path / f"{prefix}_bkg_prior.h5"

    # get_local_bkg_model returns expected RATE of bkg events / sec in
    # each CDS bin during burst
    bkg_model = get_local_bkg_model(bkg_model_path,
                                    prior_background_path)

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
        '''
        mapper.plot_ts(m_llrs, m_pix,
                       skycoord = true_src_loc,
                       grid_lines = False,
                       plot_zenith = False,
                       dpi = 300,
                       save_plot = True,
                       save_dir = output_path,
                       save_name = f"{prefix}_map.png")
        '''

        n_live_pix = save_moc_map(m_llrs, m_pix,
                                  out_nside = 64,
                                  save_dir = output_path,
                                  save_name = f"{prefix}_map")

        imax = np.argmax(m_llrs)
        pmax = m_pix[imax]

        err = moc_angular_error(pmax, true_src_loc)

        max_llr = np.max(m_llrs)
        m_probs = np.exp(m_llrs - max_llr)
        m_probs = m_probs / np.sum(m_probs) # normalize to sum to 1

        exp_angdist = moc_expected_angdist(np.pi/2 - true_src_loc.b.rad, # src colatitude
                                           true_src_loc.l.rad,           # src longitude
                                           m_pix, m_probs)
        exp_angdist = np.rad2deg(exp_angdist)

        print(f"{n_events},{transient_len:.1f},{transient_id},{n_live_pix},{err:.3f},{exp_angdist:.3f},{t:.3f}", flush=True)
