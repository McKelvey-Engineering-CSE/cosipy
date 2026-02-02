from pathlib import Path
import os

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from threeML import Powerlaw

from histpy import Histogram

from cosipy import test_data
from cosipy import FastTSMap, MOCTSMap, SpacecraftFile
from cosipy.response import FullDetectorResponse, GalacticResponse
from cosipy.response.functions import get_integrated_spectral_model

def test_ts_fit():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_full_detector_response.h5"

    orientation_path = test_data.path / "20280301_2s.fits"
    ori = SpacecraftFile.open(orientation_path)

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
                       [134.70412297, 0.,           0.,
                        137.91480599, 134.05715839, 133.01973443,
                        0.,           0.,           134.89665176,
                        0.,           0.,           135.37111622])


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
                       [40.18628386, 0.,          0.,
                        39.8845985,  40.20132198, 39.86762315,
                        0.,          0.,          40.54884861,
                        0.,          0.,          39.99131767])

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
    ori = SpacecraftFile.open(orientation_path)

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

    # test default top-k strategy
    ts_results = ts.fit(max_nside = 2,
                        data = src_bkg,
                        bkg_model = bkg,
                        spectral_flux = spectral_flux,
                        cpu_cores = 1)

    ts_values, pixels = ts_results

    assert all(pixels == [
        5,  6,  10, 11, 16,
        28, 32, 36, 48, 52,
        56, 60, 17, 29, 33,
        37, 49, 53, 57, 61,
        18, 30, 34, 38, 50,
        54, 58, 62, 19, 31,
        35, 39, 51, 55, 59,
        63
    ])

    assert np.allclose(ts_values, [
        0.,          0.,          0.,          0.,          40.31750178,
        39.78630473, 40.39347595, 40.10805456, 40.14551663, 0.,
        0.,          40.07420191, 40.07720833, 40.19657905, 40.41047826,
        0.,          40.28032052, 0.,          0.,          39.90376762,
        40.20425492, 39.07431592, 40.3217545,  40.19353339, 40.41652632,
        0.,          0.,          40.09740223, 39.81314166, 39.65452286,
        39.940159,   39.61014067, 40.65108076, 0.,          0.,
        40.2989865
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
        5,  6,  10, 11, 13,
        14, 16, 28, 32, 36,
        48, 60, 17, 29, 33,
        37, 49, 61, 18, 30,
        34, 38, 50, 62, 19,
        31, 35, 39, 51, 63
    ])

    assert np.allclose(ts_values, [
        0.,          0.,          0.,          0.,          0.,
        0.,          40.31750178, 39.78630473, 40.39347595, 40.10805456,
        40.14551663, 40.07420191, 40.07720833, 40.19657905, 40.41047826,
        0.,          40.28032052, 39.90376762, 40.20425492, 39.07431592,
        40.3217545,  40.19353339, 40.41652632, 40.09740223, 39.81314166,
        39.65452286, 39.940159,   39.61014067, 40.65108076, 40.2989865
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
        16, 20, 24, 28, 32,
        36, 40, 44, 48, 52,
        56, 60, 17, 21, 25,
        29, 33, 37, 41, 45,
        49, 53, 57, 61, 18,
        22, 26, 30, 34, 38,
        42, 46, 50, 54, 58,
        62, 19, 23, 27, 31,
        35, 39, 43, 47, 51,
        55, 59, 63
    ])

    assert np.allclose(ts_values, [
        40.31750178,  0.,          0.,           39.78630473, 40.39347595,
        40.10805456,  0.,          0.,           40.14551663, 0.,
        0.,           40.07420191, 40.07720833,  0.,          0.,
        40.19657905,  40.41047826, 0.,           0.,          39.28060584,
        40.28032052,  0.,          0.,           39.90376762, 40.20425492,
        0.,           0.,          39.07431592,  40.3217545,  40.19353339,
        0.,           0.,          40.41652632,  0.,          0.,
        40.09740223,  39.81314166, 0.,           0.,          39.65452286,
        39.940159,    39.61014067, 0.,           0.,          40.65108076,
        0.,           0.,          40.2989865
    ])
