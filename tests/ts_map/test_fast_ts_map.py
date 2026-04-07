from pathlib import Path
import os

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from threeML import Powerlaw

from histpy import Histogram

from cosipy import test_data
from cosipy import FastTSMap, MOCTSMap, SpacecraftHistory
from cosipy.response import FullDetectorResponse, GalacticResponse
from cosipy.response.functions import get_integrated_spectral_model

def test_ts_fit():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_full_detector_response.h5"

    orientation_path = test_data.path / "20280301_2s.fits"
    ori = SpacecraftHistory.open(orientation_path)

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    response = FullDetectorResponse.open(response_path)
    ts = FastTSMap(response = response,
                   orientation = ori,
                   cds_frame = "local",
                   max_cache_size = 10)

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    spectral_flux = get_integrated_spectral_model(spectrum,
                                                  response.axes["Ei"])

    ts_results = ts.fit(nside = 1,
                        data = src_bkg,
                        bkg_model = bkg,
                        spectral_flux = spectral_flux)

    assert np.allclose(ts_results,
                       [0.,           0.,           132.08983654,
                        137.91480598, 134.05715839, 0.,
                        131.2548504,  132.21155318, 134.89665176,
                        134.50732839, 135.50622357, 135.37111623])

    ts = FastTSMap(response = response,
                   orientation = ori,
                   cds_frame = "local",
                   energy_channel = [2,3],
                   max_cache_size = 10)


    ts_results = ts.fit(nside = 1,
                        data = src_bkg,
                        bkg_model = bkg,
                        spectral_flux = spectral_flux,
                        cpu_cores = 1)

    assert np.allclose(ts_results,
                       [0.,          0.,         37.4339627,
                        39.88459849, 40.20132198,  0.,
                        37.2327797,  37.4506428,  40.54884861,
                        39.69773074, 38.83421249, 39.99131767])

    ts.plot_ts(ts_results,
               skycoord = SkyCoord(l=0, b=0, unit=u.deg, frame="galactic"))

    ts.plot_ts(ts_results, containment = 0.9, save_plot = True,
               save_dir = "", save_name = "ts_map.png")

    assert Path("ts_map.png").exists()

    os.remove("ts_map.png")

def test_ts_fit_galactic():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_precomputed_response.h5"

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    response = GalacticResponse.open(response_path)
    ts = FastTSMap(response = response,
                   orientation = None,
                   cds_frame = "galactic",
                   energy_channel = [2,3])

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    spectral_flux = get_integrated_spectral_model(spectrum,
                                                  response.axes["Ei"])

    ts_results = ts.fit(nside = 1,
                        data = src_bkg,
                        bkg_model = bkg,
                        spectral_flux = spectral_flux)

    assert np.allclose(ts_results,
                       [39.75648143, 39.61688953, 39.33241148,
                        39.40114511, 39.23694982, 39.23751789,
                        39.17731599, 38.75229555, 37.41499832,
                        37.06397938, 37.03113461, 37.18839154])

def test_moc_ts_fit():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_full_detector_response.h5"

    orientation_path = test_data.path / "20280301_2s.fits"
    ori = SpacecraftHistory.open(orientation_path)

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    response = FullDetectorResponse.open(response_path)
    ts = MOCTSMap(response = response,
                  orientation = ori,
                  cds_frame = "local",
                  energy_channel = [2,3])

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    spectral_flux = get_integrated_spectral_model(spectrum,
                                                  response.axes["Ei"])

    # test top-k strategy.  Note that this test can fail if the
    # "top-k" method breaks ties differently than it did when the
    # test-case output was originally collected!  So this test might
    # not be very reproducible, in contrast to the threshold-based
    # tests below.
    ts_results = ts.fit(max_nside = 2,
                        data = src_bkg,
                        bkg_model = bkg,
                        spectral_flux = spectral_flux,
                        cpu_cores = 1,
                        strategy=MOCTSMap.TopKStrategy(8))

    ts_values, pixels = ts_results

    assert all(pixels == [
        4,  5,  9, 10, 24, 28,
        32, 44, 48, 52, 56, 60,
        25, 29, 33, 45, 49, 53,
        57, 61, 26, 30, 34, 46,
        50, 54, 58, 62, 27, 31,
        35, 47, 51, 55, 59, 63
    ])

    assert np.allclose(ts_values, [
        0.,          0.,          0.,         37.2327797,  37.39229509, 39.78630473,
        40.39347596, 38.86985166, 40.14551663, 39.92706709, 39.19653532, 40.07420192,
        37.46243839, 40.19657905, 40.41047825, 39.28060583, 40.2803205,  39.83138091,
        39.25707762, 39.90376762, 37.35958207, 39.07431591, 40.3217545,  37.24284051,
        40.41652632, 39.61022258, 39.22641583, 40.09740223, 37.43747447, 39.65452285,
        39.940159,   37.35636565,  0.,         39.92533792, 37.24385822, 40.2989865
    ])

    ts.plot_ts(*ts_results,
               skycoord = SkyCoord(l=0, b=0, unit=u.deg, frame="galactic"))

    ts.plot_ts(*ts_results, containment = 0.9, save_plot = True,
               save_dir = "", save_name = "ts_map.png")

    assert Path("ts_map.png").exists()

    os.remove("ts_map.png")

    # test containment strategy
    ts_results = ts.fit(max_nside = 2,
                        data = src_bkg,
                        bkg_model = bkg,
                        spectral_flux = spectral_flux,
                        strategy=MOCTSMap.ContainmentStrategy(0.9))

    ts_values, pixels = ts_results

    assert all(pixels == [
        4,  5,  9, 24, 28, 32,
        40, 44, 48, 52, 56, 60,
        25, 29, 33, 41, 45, 49,
        53, 57, 61, 26, 30, 34,
        42, 46, 50, 54, 58, 62,
        27, 31, 35, 43, 47, 51,
        55, 59, 63
    ])

    assert np.allclose(ts_values, [
        0.,          0.,          0.,          37.39229509, 39.78630473, 40.39347596,
        38.79974817, 38.86985166, 40.14551663, 39.92706709, 39.19653532, 40.07420192,
        37.46243839, 40.19657905, 40.41047825, 37.23577505, 39.28060583, 40.2803205,
        39.83138091, 39.25707762, 39.90376762, 37.35958207, 39.07431591, 40.3217545,
        38.98305396, 37.24284051, 40.41652632, 39.61022258, 39.22641583, 40.09740223,
        37.43747447, 39.65452285, 39.940159,   37.24223294, 37.35636565,  0.,
        39.92533792, 37.24385822, 40.2989865
    ])

    # test padding strategy over a different containment threshold
    # (yields same result as previous test)
    ts_results = ts.fit(max_nside = 2,
                        data = src_bkg,
                        bkg_model = bkg,
                        spectral_flux = spectral_flux,
                        strategy=MOCTSMap.PaddingStrategy(
                            MOCTSMap.ContainmentStrategy(0.5)))

    ts_values, pixels = ts_results

    assert all(pixels == [
        16, 20, 24, 28, 32, 36,
        40, 44, 48, 52, 56, 60,
        17, 21, 25, 29, 33, 37,
        41, 45, 49, 53, 57, 61,
        18, 22, 26, 30, 34, 38,
        42, 46, 50, 54, 58, 62,
        19, 23, 27, 31, 35, 39,
        43, 47, 51, 55, 59, 63
    ])

    assert np.allclose(ts_values, [
        0.,          0.,          37.39229509, 39.78630473, 40.39347596,  0.,
        38.79974817, 38.86985166, 40.14551663, 39.92706709, 39.19653532,  40.07420192,
        0.,          0.,          37.46243839, 40.19657905, 40.41047825,  0.,
        37.23577505, 39.28060583, 40.2803205,  39.83138091, 39.25707762,  39.90376762,
        0.,          0.,          37.35958207, 39.07431591, 40.3217545,   0.,
        38.98305396, 37.24284051, 40.41652632, 39.61022258, 39.22641583,  40.09740223,
        0.,          0.,          37.43747447, 39.65452285, 39.940159,    0.,
        37.24223294, 37.35636565, 0.,          39.92533792, 37.24385822,  40.2989865
    ])
