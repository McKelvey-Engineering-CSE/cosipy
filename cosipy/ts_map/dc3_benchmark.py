#
# BENCHMARK SCRIPT FOR COSI DC3 EXAMPLE BURST
#

from pathlib import Path
from time import time
import sys
from argparse import ArgumentParser

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from threeML import Powerlaw

from histpy import Histogram

from cosipy import SpacecraftHistory, FastTSMap, MOCTSMap
from cosipy.util import fetch_wasabi_file
from cosipy.response import FullDetectorResponse
from cosipy.response.functions import get_integrated_spectral_model

from dc3_benchmark_cosipy import FastTSMap_Cosipy


parser = ArgumentParser(description="DC3 benchmark script")

# positional arguments
parser.add_argument("mode", choices=["cosipy_interp",
                                     "cosipy_numba",
                                     "emsoft_outmem",
                                     "emsoft_inmem",
                                     "emsoft_moc"],
                    help="which implementation of mapping to run")

parser.add_argument("-d", "--data-path", type=Path,
                    default=Path('.'),
                    help="path to benchmark input files")

args = vars(parser.parse_args(sys.argv[1:]))

#
# It is possible to obtain the data files used by this benchmark
# directly from COSI's public server using the (currently
# commented-out) calls to fetch_wasabi_file below.  But for
# convenience, we should provide these files as part of the artifact.
#

data_dir = args["data_path"]

GRB_signal_path = data_dir/"grb_binned_data.hdf5"

# download GRB signal file ~76.90 KB
#if not GRB_signal_path.exists():
#    fetch_wasabi_file("COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_binned_data.hdf5", GRB_signal_path)

background_path = data_dir/"bkg_binned_data_local.hdf5"

# download background file ~255.97 MB
#if not background_path.exists():
#    fetch_wasabi_file("COSI-SMEX/cosipy_tutorials/ts_maps/bkg_binned_data_local.hdf5", background_path)

orientation_path = data_dir/"20280301_3_month_with_orbital_info.fits"

# download orientation file ~684.38 MB
#if not orientation_path.exists():
#    fetch_wasabi_file("COSI-SMEX/DC3/Data/Orientation/20280301_3_month_with_orbital_info.fits", orientation_path)

response_path = data_dir/"SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5"

# download response file ~839.62 MB
#if not response_path.exists():
#    fetch_wasabi_file("COSI-SMEX/cosipy_tutorials/Data/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.h5")

index = -2.2
K = 10 / u.cm / u.cm / u.s / u.keV
piv = 100 * u.keV
spectrum = Powerlaw()
spectrum.index.value = index
spectrum.K.value = K.value
spectrum.piv.value = piv.value
spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

# Read the GRB signal
signal = Histogram.open(GRB_signal_path)

# get the starting and ending time tag of the GRB
grb_tmin = signal.axes["Time"].edges.min()
grb_tmax = signal.axes["Time"].edges.max()

# project to three axes: measure energy(Em), scattering
# direction(PsiChi) and Compton scattering angle (Phi)
signal = signal.project(['Em', 'Phi', 'PsiChi'])

# load the background file

bkg_full = Histogram.open(background_path)

bkg_times = bkg_full.axes['Time'].edges.value

bkg_tmin_idx = np.searchsorted(bkg_times, grb_tmin.value, side='left')
bkg_tmax_idx = np.searchsorted(bkg_times, grb_tmax.value, side='right')
bkg = bkg_full.slice[bkg_tmin_idx:bkg_tmax_idx,:]  # It slices the Time axis

# project to three axes: measure energy(Em), scattering
# direction(PsiChi) and Compton scattering angle (Phi)
bkg = bkg.project(['Em', 'Phi', 'PsiChi'])

# assemble the data
data = bkg + signal

# calculate the duration of the background
bkg_full_duration = (bkg_full.axes['Time'].edges.max() - bkg_full.axes['Time'].edges.min())

# average the background model down to 40s
bkg_model = bkg_full/(bkg_full_duration/40)

# project to three axes: measure energy(Em), scattering
# direction(PsiChi) and Compton scattering angle (Phi)
bkg_model = bkg_model.project(['Em', 'Phi', 'PsiChi'])

# read the full oritation but only get the interval for the GRB
grb_ori = SpacecraftHistory.open(orientation_path,
                                 tstart = Time(grb_tmin, format = "unix"),
                                 tstop = Time(grb_tmax, format = "unix"))

# clear redundant data from RAM
del bkg_full


if args["mode"].startswith("cosipy"):

    is_cosipy = True
    normfit_mode = args["mode"].removeprefix("cosipy_")

    # here let's create a FastTSMap object for fitting the ts map in the
    # following cells
    ts = FastTSMap_Cosipy(data = data,
                          bkg_model = bkg_model,
                          orientation = grb_ori,
                          response_path = response_path,
                          cds_frame = "local")

else: # an emsoft mode

    is_cosipy = False

    # unlike cosipy, we use float32 for performance
    data = data.astype(np.float32)
    bkg_model = bkg_model.astype(np.float32)

    response = FullDetectorResponse.open(response_path)

    spectral_flux = get_integrated_spectral_model(spectrum,
                                                  response.axes["Ei"])

    is_moc = False

    match args["mode"]:

        case "emsoft_inmem": # response in memory

            ts = FastTSMap(orientation = grb_ori,
                           response = response,
                           cds_frame = "local",
                           response_in_memory=True)

        case "emsoft_outmem": # response not in memory

            ts = FastTSMap(orientation = grb_ori,
                           response = response,
                           cds_frame = "local",
                           response_in_memory=False)

        case "emsoft_moc": # emsoft version, MOC map (in memory)

            is_moc = True
            ts = MOCTSMap(orientation = grb_ori,
                          response = response,
                          cds_frame = "local",
                          response_in_memory=True)


NTRIALS = 5

t_total = 0.
for i in range(NTRIALS + 1):

    t_start = time()

    if is_cosipy:
        ts_results = ts.fit(nside = 16,
                            energy_channel=[0,10],
                            spectrum = spectrum,
                            cpu_cores = 8,
                            fast_norm_opt=normfit_mode)
    else: # emsoft version
        if not is_moc:
            ts_results = ts.fit(data = data,
                                bkg_model = bkg_model,
                                spectral_flux = spectral_flux,
                                nside = 16,
                                cpu_cores = 8)
        else:
            ts_results = ts.fit(data = data,
                                bkg_model = bkg_model,
                                spectral_flux = spectral_flux,
                                max_nside = 16,
                                cpu_cores = 8)

    t_end = time()

    if i > 0: # skip warmup trial
        t_total += t_end - t_start


if NTRIALS > 0:
    print(f"TIME: {t_total/NTRIALS:.3f} s")

# This the true location of the GRB
#coord = SkyCoord(l = 93, b = -53, unit = (u.deg, u.deg), frame = "galactic")
