#
# USAGE:
# map_adapt_transients <burst_path> <output_path>
#
#  burst_path : where to read burst data from
#  output_path : where to write output maps

from pathlib import Path
import time
import sys

import numpy as np
from scipy.stats import chi2

from astropy.table import Table
import astropy.units as u

import mhealpy as hp

import h5py as h5
import hdf5plugin # for compressed response

from histpy import Histogram

from cosipy import FastTSMap, MOCTSMap
from cosipy.response import GalacticResponse
from cosipy.response.functions import get_integrated_spectral_model

from cosipy.ts_map.read_dc3_data import (
    read_transient_params,
    read_unbinned_events,
    combine_unbinned_events,
)

from cosipy.ts_map.mapping_time import trim_events

from sphdist import moc_expected_angdist

def moc_llr_to_prob(llrs, uniq_pix):

    # Let i* be the ML pixel. We want to compute
    # Pr(src in i | data) for each pixel i.
    #
    # We get LLR(i) =
    #  log( Pr(data | src in i) / Pr(data | bg) )
    #
    # So we can compute
    #
    # Pr(src in i | data)
    #   propto Pr(data | src in i) * Pr(src in i)
    #   propto exp(LLR(i)) * Pr(src in i)
    #   propto exp(LLR(i) - LLR(i*)) * Pr(src in i)
    #
    # and we can normalize the last term to sum to over all pixels.
    # The -LLR(i*) prevents us from working with very small numbers.

    max_llr = np.max(llrs)
    scaled_likelihood = np.exp(llrs - max_llr)

    # prior Pr(src in i) is proportional to the fraction
    # of the sphere occupied by pixel i.  In a multires
    # map, these priors need not be the same for all pixels!
    prior_prob = 1. / hp.nside2npix(hp.uniq2nside(uniq_pix))

    probs = scaled_likelihood * prior_prob
    probs /= np.sum(probs) # normalize to sum to 1

    return probs

def save_moc_map(llrs, uniq_pix, out_nside,
                 true_src_loc, n_src_events, n_bkg_events,
                 save_name, save_dir):

    probs = moc_llr_to_prob(llrs, uniq_pix)

    # Save only the 99.9% confidence region, i.e.,
    # all pixels whose llr score is no more than
    # max_diff below the max llr
    max_diff = chi2.ppf(0.999, df=2)
    save_mask = (llrs >= np.max(llrs) - max_diff)

    uniq_pix = uniq_pix[save_mask]
    probs    = probs[save_mask]

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

    lons, lats = hp.pix2ang(out_nside, pix, nest=True, lonlat=True)

    with h5.File(save_dir / (save_name + ".h5"), "w") as f:
        f.attrs["source"] = np.array([true_src_loc.b.deg, true_src_loc.l.deg],
                                     dtype=np.float32)
        f.attrs["n_src_events"] = n_src_events
        f.attrs["n_bkg_events"] = n_bkg_events

        f.create_dataset("pixel",       data=pix,
                         dtype=np.int32,
                         compression=hdf5plugin.Bitshuffle())
        f.create_dataset("latitude",    data=lats,
                         dtype=np.float32,
                         compression=hdf5plugin.Bitshuffle())
        f.create_dataset("longitude",   data=lons,
                         dtype=np.float32,
                         compression=hdf5plugin.Bitshuffle())
        f.create_dataset("probability", data=probs,
                         dtype=np.float32,
                         compression=hdf5plugin.Bitshuffle())


transient_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

output_path = output_dir
output_path.mkdir(parents=True, exist_ok=True)

model_dir = Path("/project/cassini/adapt_grbs")

bkg_model_path = model_dir / "adapt_bkg_model.h5"

response_path = model_dir / "adapt_response_w_area.h5"

num_cpus = 8

def angular_error(nside, ml_pix, true_src_loc):

    ml_center = np.array(hp.pix2vec(ipix=ml_pix, nside=nside, nest=True))
    src_loc   = true_src_loc.cartesian.xyz.value # unit vector

    best_src_dist = np.rad2deg(np.arccos(np.dot(ml_center, src_loc)))

    return best_src_dist

def moc_angular_error(ml_pix, true_src_loc):

    ml_nside, ml_nest = hp.uniq2nest(ml_pix)

    ml_center = np.array(hp.pix2vec(ipix=ml_nest, nside=ml_nside, nest=True))
    src_loc   = true_src_loc.cartesian.xyz.value # unit vector

    best_src_dist = np.rad2deg(np.arccos(np.dot(ml_center, src_loc)))

    return best_src_dist

def get_bkg_prior_rate(bkg_path):
    with h5.File(bkg_path, "r") as f:
        return f.attrs["bg_prior"]

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
sources.sort(reverse=True)

# Run a few warmup iterations to make sure the JIT runs and avoid
# other startup transients.  Empirically, 5 is the minimum number of
# iterations needed on ARM to avoid seeing an artificially long
# running time for the first recorded iteration.
n_warmup = 3
sources = [sources[0]]*n_warmup + sources

print("fluence,length,alt,az,transient_id,est_length,n_src,n_bkg,alt_ml,az_ml,err,exp_angdist,conf,wait_time,mapping_time", flush=True)

results = []
for i, signal_path in enumerate(sources):

    # extract source info from name
    # expected name: adapt_{fluence}_{length}_p{polar}_a{azimuthal}_{inst}_...
    fields = signal_path.name.split("_")
    inst = fields[5]

    prefix = "_".join(fields[:6])
    params_path = transient_path / f"{prefix}_params.txt"

    # WARNING: do not use any of these values (except maybe ts)
    # in mapping! Doing so is "cheating"
    _, true_src_loc, (ts, true_te), fluence = \
        read_transient_params(params_path)

    background_path = transient_path / f"{prefix}_background.h5"

    # get estimate of background rate from prior data
    bkg_rate = get_bkg_prior_rate(background_path)

    # scale background model to estimated rate
    bkg_model = Histogram.open(bkg_model_path)
    bkg_model = bkg_model.project(("Em", "Phi", "PsiChi"))
    bkg_model *= bkg_rate

    # retrieve and combine the unbinned events
    signal_events = read_unbinned_events(signal_path)
    bkg_events = read_unbinned_events(background_path)
    events = combine_unbinned_events(signal_events, bkg_events)


    # we leave compute time to determine te outside our timing
    # region, assuming that cost to determine it is negligible
    # compared to the wait time we incur before we choose it

    '''
    # CHEAT: keep only events during the transient
    te = true_te
    wait_time = 0

    e_end = np.searchsorted(events["time"].value, te, side='right')
    events["time"] = events["time"][:e_end]
    events["Em"]   = events["Em"][:e_end]
    events["Phi"]  = events["Phi"][:e_end]
    events["PsiChi"] = events["PsiChi"][:,:e_end]
    '''

    te, wait_time = trim_events(events, ts, bkg_rate)

    timer_start = time.time()

    # compute total expected bg fluence during transient
    # based on estimated end time
    bkg_model *= te - ts

    # compute (rough!) Ei spectral flux approximation as a histogram of the
    # Em values in the observed events
    spectral_flux, _ = np.histogram(events["Em"],
                                    bins = response.axes["Em"].edges)
    spectral_flux = Histogram(response.axes["Em"],
                              spectral_flux,
                              unit=1/(u.s * u.cm**2),
                              dtype=np.float32)

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
    timer_end = time.time()

    mapping_time = timer_end - timer_start

    if i >= n_warmup:

        # deduce actual numbers of signal and bg events in transient
        # -- requires knowledge of at least one of signal, bg labels
        src_mask = (signal_events["time"] <= te * u.s)
        n_src = np.count_nonzero(src_mask)
        n_bkg = len(events["time"]) - n_src

        '''
        mapper.plot_ts(m_llrs, m_pix,
                       skycoord = true_src_loc,
                       grid_lines = True,
                       plot_zenith = False,
                       dpi = 300,
                       save_plot = True,
                       save_dir = output_path,
                       save_name = f"{prefix}_map.png")

        save_moc_map(m_llrs, m_pix,
                    out_nside = 64,
                     true_src_loc = true_src_loc,
                     n_src_events = n_src,
                     n_bkg_events = n_bkg,
                     save_dir = output_path,
                     save_name = f"{prefix}_map")
        '''


        # maximum likelihood pixel and LLR
        ml_idx = np.argmax(m_llrs)
        ml_pix = m_pix[ml_idx]
        ml_llr = m_llrs[ml_idx]

        # source pixel and LLR
        b = hp.HealpixBase(uniq=m_pix, scheme="NUNIQ", coordsys="G")
        src_pix = b.vec2pix(*true_src_loc.cartesian.xyz.value)
        src_llr = m_llrs[src_pix]

        # angular distance from ML pixel center to true source
        err = moc_angular_error(ml_pix, true_src_loc)

        # expected angular distance from *all* pixels to true source
        m_probs = moc_llr_to_prob(m_llrs, m_pix)
        exp_angdist = moc_expected_angdist(np.pi/2 - true_src_loc.b.rad, # src colatitude
                                           true_src_loc.l.rad,           # src longitude
                                           m_pix, m_probs)
        exp_angdist = np.rad2deg(exp_angdist)

        # confidence (containment) score of source pixel
        conf = chi2.cdf(np.max(m_llrs) - src_llr, df=2)

        ml_az, ml_alt = b.pix2ang(ml_idx, lonlat=True)
        true_alt = true_src_loc.b.deg
        true_az  = true_src_loc.l.deg

        print(f"{fluence:.1f},{(true_te - ts):.1f},{true_alt:.3f},{true_az:.3f},{inst},{(te - ts):.1f},{n_src},{n_bkg},{ml_alt:.3f},{ml_az:.3f},{err:.3f},{exp_angdist:.3f},{conf:.4f},{wait_time:.3f},{mapping_time:.3f}", flush=True)
