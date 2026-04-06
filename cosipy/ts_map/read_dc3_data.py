import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

import h5py

from histpy import Histogram

from scoords import Attitude
from astromodels import Band_grbm, Powerlaw, Cutoff_powerlaw_Ep

def read_transient_params(params_path):
    """
    Read the true source location and spectral parameters of a burst
    (i.e., the ground truth) from the parameters file that we extracted
    from the DC3 parameter data.

    Params file format is two lines:
      "location" lat lon (in degrees)
      "spectrum" <spectral type and parameters>

    Parameters
    ----------
    params_path -- path
      path to parameters file

    Returns
    -------
    tuple of (spectrum, true location, src_ival, fluence)
        - spectrum is an astromodels Spectrum object
        - true location is an astropy SkyCoord
        - src_ival is a pair (start_time, end_time) for burst interval
        - fluence is a floating-point value -- total fluence of burst
    """

    with open(params_path, "r") as params:
        locline = params.readline().strip().split(" ")
        specline = params.readline().strip().split(" ")
        light_curve = params.readline().strip().split(" ")

        true_src_loc = SkyCoord(b = float(locline[1]),
                                l = float(locline[2]),
                                unit = u.deg,
                                frame = "galactic")

        spectype = specline[1]

        if spectype == "Band":
            spec_params = {
                'piv':   100,
                'alpha': float(specline[4]),
                'beta':  float(specline[5]),
                'xc':    float(specline[6]),
            }
        elif spectype == "PowerLaw":
            spec_params = {
                'piv': float(specline[2]),
                'alpha': float(specline[4])
            }
        elif spectype == "Comptonized":
            spec_params = {
                'piv':   100,
                'alpha': float(specline[4]),
                'xp':    float(specline[5])
            }
        else:
            raise ValueError(f"Unsupported spectrum type {spectype}")

        if spectype == "Band":
            spectrum = Band_grbm() # matches MegaLib definition
            spectrum.alpha.value = spec_params["alpha"]
            spectrum.beta.value = spec_params["beta"]
            spectrum.xc.value = spec_params["xc"]
        elif spectype == "PowerLaw": # is this right?
            spectrum = Powerlaw()
            spectrum.index.value = -spec_params['alpha']
        elif spectype == "Comptonized": # matches MegaLib definition
            spectrum = Cutoff_powerlaw_Ep()
            spectrum.index.value = spec_params['alpha']
            spectrum.xp.value = spec_params['xp']
            spectrum.xp.unit = u.keV

        # not necessary to provide accurate K -- intensity
        # is estimated as part of fitting
        spectrum.K.value = 1
        spectrum.K.unit = 1/(u.cm**2 * u.s * u.keV)
        spectrum.piv.value = spec_params["piv"]
        spectrum.piv.unit = u.keV

        if len(light_curve) < 4:
            fluence = 0.
            src_ival = (0., 0.)
        else:
            # read light curve info for transient
            ts      = float(light_curve[1])
            src_len = float(light_curve[2])
            fluence = float(light_curve[3])
            src_ival = (ts, ts + src_len)

        return spectrum, true_src_loc, src_ival, fluence


def read_unbinned_events(data_path, max_events = None):
    """
    Read unbinned event data set and store them in a dictionary.  The
    unbinned data has the following components in time order:

    "time" -- event time (unit: seconds)
    "Em"   -- measured energy (unit: keV)
    "Phi"  -- opening angle of Compton ring (unit: deg)
    "PsiChi" -- center vector of Compton ring (unit: deg)

    PsiChi is an 2 x N array A, where A[0] is lon (corresponding to Chi)
    and A[1] is lat (corresponding to Psi) in the spacecraft frame.

    We assume that the input HDF5 files contain time, Em, Phi, and
    separate Psi and Ch, all of which are already in time order. The
    last three are all in radians and need to be converted to degrees,
    and the co-latitude Psi needs to be converted to latitude.

    Parameters
    ----------
      data_path -- path to HDF5 file containing event data
      max_events -- maximum number of events to return; if data set
        contains more events, the subset returned is selected randomly

    Returns
    -------
      dict with contents as above

    """

    data = h5py.File(data_path)

    # read event data
    events = { field : np.array(data[field])
               for field in ("time", "Em", "Phi", "Psi", "Chi") }

    events["time"] = events["time"].astype(np.float64)
    if "time_offset" in data.attrs:
        events["time"] += data.attrs["time_offset"]

    data.close()

    # Convert Phi to degrees
    events["Phi"] = np.rad2deg(events["Phi"])

    # merge Psi and Chi values into a single PsiChi array.
    # psichi[0] is longitude, while psichi[1] is latitude
    # (both in degrees)
    events["PsiChi"] = np.vstack((np.rad2deg(events["Chi"]),
                                  90 - np.rad2deg(events["Psi"])))
    del events["Psi"]
    del events["Chi"]

    # add units to data for compatibility with response axes later on
    for key, unit in zip(("time", "Em", "Phi", "PsiChi"),
                         (u.s, u.keV, u.deg, u.deg)):
        events[key] = u.Quantity(events[key], unit=unit, copy=False)

    # if requested, sub-sample the input set of events
    n_events = len(events["time"])
    if max_events is not None:
        if isinstance(max_events, float): # sample fraction of input
            assert max_events <= 1.0
            max_events = int(max_events * n_events)

        if max_events < n_events:
            selection = np.sort(np.random.choice(n_events, size=max_events, replace=False))

            for key in ("time", "Em", "Phi"):
                events[key] = events[key][selection]
            events["PsiChi"] = events["PsiChi"][:,selection]

    return events


def combine_unbinned_events(e1, e2):
    """
    Combine two unbinned event dictionaries read by read_unbinned_events().
    The result is in time order for each combined field.

    Parameters
    ----------
    e1, e2 -- event dictionaries

    Returns
    -------
    combined dictionary

    """

    # determine time ordering of all events and sort times together
    time = np.hstack((e1["time"], e2["time"]))
    order = np.argsort(time)
    events = { "time" : time[order] }

    # sort together other event data by time. Fields may be 1D or 2D;
    # for 2D arrays, we want to sort the columns of every row.
    for field in ("Em", "Phi", "PsiChi"):
        events[field] = np.hstack((e1[field], e2[field]))[..., order]

    return events


def get_local_bkg_model(bkg_model_path,
                        local_background_path):
    """
    Get a model of the local background rates.  We start with a general
    background CDS model read from a file and then scale its intensity
    to match a local estimated of the background intensity near the GRB.

    Parameters
    ----------
    bkg_model_path : path
      path to general bkg model, which is a Histogram containing
      *relative* event rates for each CDS bin; rates add to 1 over
      the whole bin.
    local_background_path : path
      path to HDF5 file containing a slice of the background from
      a defined period prior to the GRB.

    Returns
    -------
    Histogram giving expected numbers of events per second in each
    CDS bin during the GRB.

    """

    # estimate the background event rate from the period prior to the GRB
    with h5py.File(local_background_path) as local_bkg:
        t = local_bkg["time"]
        ts = t[0]
        te = t[-1]
        bkg_rate = len(t) / (te - ts)

    bkg_model = Histogram.open(bkg_model_path)
    bkg_model = bkg_model.project(("Em", "Phi", "PsiChi"))

    # scale the background model by the expected event count during the GRB
    return bkg_model * bkg_rate
