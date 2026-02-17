from pathlib import Path
import sys

import numpy as np

import h5py as h5

from astropy.time import Time
import astropy.units as u

from mhealpy import HealpixBase

from cosipy.spacecraftfile import SpacecraftFile
from cosipy.response import FullDetectorResponse

from cosipy.ts_map.read_dc3_data import read_burst_params

def nd_sample(P, nSamples=1):
    """Randomly sample nSamples indices from an n-dimensional
    distribution. The probability of picking a given index
    [i_1, ..., i_k] is given by P[i_1, ..., i_k].

    Inputs
    ------
    P : n-dimensional ndarray of float
      A probability distribution over the n-dimensional indices of P
    nSamples : int (optional)
      Number of indices to sample (default = 1)

    Returns
    -------
    A P.ndim-tuple of ndarrays of int giving the sampled indices

    """

    P_f = np.ravel(P)
    bins = np.random.choice(len(P_f), size=nSamples, p=P_f)
    return np.unravel_index(bins, P.shape)


def sample_source_events(nSamples, source_fluence, ei_flux, psrs):
    """Sample a set of observed photon events from a source using the
    response model.

    Events are sampled assuming that the the source follows a
    discretized path of local source directions, spending a "run" of
    time pointed in each direction. The relative fluence during each
    run (which depends on both the source's light curve and the length
    of the run) is specified by the array 'source_fluence'.  The
    expected flux of incident events in each energy bin (which is
    time-invariant) is specified by the array 'ei_flux'.

    Note that source_fluence and ei_flux are used only for their
    relative values in each run/bin; they need not be normalized.

    For each sample, we choose the observed event's run and incident
    energy bin using source_fluence, ei_flux, and the detector
    effective area model, which depends on both the run's source
    direction and the event's incident energy.  We then provide a CDS
    voxel for the event by sampling from the PSR for its run and
    incident energy bin.

    Parameters
    ----------
    nSamples : int
      Number of events to sample
    source_fuence : array of float
      Fluence of source during each run with a fixed source pixel
    ei_flux : array of float
      Flux of events in each incident energy bin
    psrs : list of pairs of arrays
      PSRs corresponding to each run. Each pair is
      an array (P, A), where
      - P is an array of shape |E_i| x CDS whose elements sum to 1
      - A is an effective area array of shape |E_i|

    Returns
    -------
    A triple of integer arrays (runs, ei_bins, cds_coords), each of
    length nSamples, describing for each sampled event its:
     - run within source path
     - incident energy bin
     - CDS voxel [an array of nSamples x CDS values]

    Events are sorted by run and incident energy bin.

    """

    # effective area per E_i bin during each run
    A = np.stack(tuple(psr[1] for psr in psrs))

    # Compute joint PDF over (run, energy bin) for observed
    # events
    D = A * np.outer(source_fluence, ei_flux)
    D /= np.sum(D)

    # sample runs / energy bins for observed events
    runs, ei_bins = nd_sample(D, nSamples)

    # sort samples by run and energy for locality of reference
    idx = np.lexsort((runs, ei_bins))
    runs = runs[idx]
    ei_bins = ei_bins[idx]

    # CDS distributions per E_i during each run
    P = np.stack(tuple(psr[0] for psr in psrs))

    # sample CDS coords for observed events, creating a list
    # of samples with both source and CDS coords
    samples_cds = []
    for run, ei_bin in zip(runs, ei_bins):
        P_CDS = P[run, ei_bin]
        cds_coords = nd_sample(P_CDS)
        samples_cds.append(np.concatenate(cds_coords))

    # return source coordinates and CDS voxels per sample
    return runs, ei_bins, np.stack(samples_cds)


def get_integrated_light_curve(light_curve, time_endpts):
    """
    Integrate a LightCurve function within each of the bins whose
    endpoints are defined by an array of times.

    Parameters
    ----------
    light_curve : LightCurve object
      A function object that describes the transient's time-dependent total
      flux
    time_endpts : array of float
      Endpoints of time bins

    Returns
    -------
    Array of float giving total fluence in each time bin

    """

    from scipy import integrate

    light_curve_binned = [
        integrate.quad(light_curve, lo_lim, hi_lim)[0]
        for lo_lim, hi_lim in zip(time_endpts[:-1],
                                  time_endpts[1:])
    ]

    return np.array(light_curve_binned)


def CDS_voxels_to_values(CDS_voxels, CDS_axes):
    """Convert an array of CDS voxels to representative values.

    Parameters
    ----------
    CDS_voxels : 2D array of int
      Each row of array specifies voxel coordinates of one event
    CDS_axes : Axes object
      Ordered set of axes describing binning for each voxel coordinate;
      voxel coordinates are assumed to match order of axes

    Returns
    -------
    2D array of float with each voxel replaced by its value, in the
    order (Em, Phi, Psi, Chi).  Here,
      - Em is measured energy (keV)
      - Phi is scattering angle (rad)
      - Psi, Chi gives PsiChi dir in polar coordinates as colat, lon
        (rad, rad)

    """

    vals = []

    Em_idx = CDS_axes.label_to_index("Em")
    vals.append(CDS_axes["Em"].centers[CDS_voxels[:,Em_idx]].value)

    Phi_idx = CDS_axes.label_to_index("Phi")
    vals.append(CDS_axes["Phi"].centers[CDS_voxels[:,Phi_idx]].value *
                np.pi/180)

    PsiChi_idx = CDS_axes.label_to_index("PsiChi")
    vals.extend(CDS_axes["PsiChi"].pix2ang(CDS_voxels[:,PsiChi_idx],
                                            lonlat=False))

    return np.column_stack(vals)


def get_source_samples(orientations, response,
                       light_curve, spectrum,
                       galactic_source,
                       nSamples):
    """
    Compute a random sample of *observed* events from a transient
    described by

      - a galactic-frame source direction
      - a light curve (d_fluence / d_time in arbitrary units)
      - an energy spectrum (d_flux / d_energy in arbitrary units)

    Sampling considers the light curve and spectrum of the transient,
    the changing orientation of the detector with time, its effective
    area vs local-frame source direction and incident photon energy, and
    the distribution of CDS parameters described by the response model.

    Sampling is discretized as follows:
      - Time is discretized into the bins defined by the record of
        spacecraft orientation.
      - Source direction is discretized onto the HEALPIX grid used
        define the response's NuLambda axis.
      - Incident photon energy is discretized according to the E_i
        bins of the response.
      - The event's distribution over CDS parameters is discretized
        according to the response's CDS voxels.

    Sampling selects a time bin, an E_i bin, and a CDS voxel for each
    observed event. After sampling, we replace the sampled time bin
    with a real-valued time chosen uniformly at random between its
    start and end; we replace the E_i bin with its center value; and
    we replace the CDS voxel with the Em/phi/psichi values
    corresponding to the voxel's center.

    (We could uniformly jitter the E_i value and CDS parameters as well,
    but that is no more accurate than picking the centers, because the
    values within one bin/voxel are not uniformly distributed. We jitter
    the times only to avoid a large number of simultaneous events.)

    Parameters
    ----------
    orientations : SpacecraftFile object
      A record of the spacecraft orientation with respect to time
    response : FullDetectorResponse object
      A description of the detector's effective area and CDS response
      as a function of local-frame source direction and incident photon
      energy
    light_curve : LightCurve object
      A function object that describes the transient's time-dependent total
      flux
    spectrum : Spectrum object
      A function object that describes how the transient's relative flux
      varies with incident energy (assumed to be time-invariant)
    galactic_source : a Cartesian 3-vector as an ndarray
      Direction from which the source comes
    nSamples : int
      Number of samples to compute

    Returns
    -------
    2D array of size nSamples x 6, containing one event per row.
    The contents of each event's row are
      - time (s)
      - incident energy E_i (keV)
      - measured energy E_m (keV)
      - scattering angle phi (rad)
      - axis psichi through the first two hits as (colat, lon) in
        local frame (rad, rad)

    """

    from cosipy.response.functions import get_integrated_spectral_model

    # extract subset of orientation history during the transient
    ori = orientations.source_interval(Time(light_curve.t_start,
                                            format="unix"),
                                       Time(light_curve.t_end,
                                            format="unix"))

    times = ori.get_time().to_value("unix")

    # compute local-frame path corresponding to source direction
    phis, thetas = ori.get_target_in_sc_frame(galactic_source)

    # compute local-frame source pixel at start of each time bin
    # FIXME: should we do the middle of each bin?
    pixels = response.axes["NuLambda"].ang2pix(theta=thetas[:-1],
                                               phi=phis[:-1],
                                               lonlat=False)

    # compute total fluence of source in each time bin.  We assume
    # that no time bin is blocked by SAA passage or earth occultation.
    light_curve_binned = get_integrated_light_curve(light_curve, times)

    # find "breakpoint" indices where source pixel changes; 0 is
    # always a breakpoint.
    bp = np.nonzero(np.diff(pixels, prepend=-1))[0]

    # compute total fluence for each contiguous run of time bins
    # in which the source pixel does not change.
    source_fluence = np.add.reduceat(light_curve_binned, bp)

    # get relative flux for each energy bin from spectrum using Ei bins
    ei_flux = get_integrated_spectral_model(spectrum,
                                            response.axes["Ei"]).contents.value

    eff_area_correction = response.eff_area_correction
    cds_axes = tuple(range(1, response.ndim - 1))

    # from response, extract PSR for each run's source pixel and
    # store it as a pair (A, P), where
    #  A gives the effective area for the source dir for each E_i
    #  P gives a probability distribution over CDS voxels for each E_i
    psrs = []
    for pix in pixels[bp]:
        counts = response.get_counts(pix)
        n_c = np.sum(counts, axis=cds_axes) # total counts per E_i

        A = eff_area_correction * n_c
        P = counts / np.expand_dims(n_c, axis=cds_axes)

        psrs.append((P,A))

    # generate source events given fluence of each run, flux of each
    # energy bin, and effective area and PSR during each run
    event_runs, event_ei_bins, events_cds_voxels = \
        sample_source_events(nSamples, source_fluence, ei_flux, psrs)

    #
    # Each generated event is assigned a run (to minimize the number
    # of duplicate PSR arrays we need to keep around at once), but we
    # want it to be assigned an individual time bin.  So we now sample
    # a time bin from each event's run, accounting for changes in
    # fluence during the run.
    #

    # normalize light curve to sum to 1 within each run
    run_len = np.diff(bp, append=len(light_curve_binned))
    bin_run = np.repeat(np.arange(len(bp)), run_len)
    ps = light_curve_binned / source_fluence[bin_run]

    # assign a time bin to each event by sampling a bin from its run,
    # weighting each bin according to its relative fluence
    time_bins = np.array([
        np.random.choice(run_len[r],
                         p = ps[bp[r]:bp[r]+run_len[r]])
        for r in event_runs
    ])

    durations = np.diff(times)
    event_times = times[time_bins] + np.random.rand(len(time_bins)) * durations[time_bins]

    # assign each event the center value of its Ei bin
    event_eis = response.axes["Ei"].centers[event_ei_bins].value

    event_cds_coords = CDS_voxels_to_values(events_cds_voxels,
                                            response.axes[2:])

    return np.column_stack((event_times, event_eis, event_cds_coords))


def write_source_events(source_samples, fname):
    """
    Write a set of sampled source events produced by get_source_samples()
    to a FITS file in the form expected by the GRB mapping code.

    Parameters
    ----------
    source_samples : nSamples x 6 array of float
      Sampled events (Ei values are not saved)
    fname : string
      name of output file to write; should end with '.fits'
    """

    dict = {
        "time" : source_samples[:,0], # seconds since UNIX epoch
        "Em"   : source_samples[:,2], # keV
        "Phi"  : source_samples[:,3], # radians
        "Psi"  : source_samples[:,4], # *co-latitude* in radians
        "Chi"  : source_samples[:,5], # radians
    }

    # sort samples by time
    time = dict['time']
    order = np.argsort(time)

    with h5.File(fname, "w") as f:
        for field in dict:
            f.create_dataset(field,
                             data=dict[field][order],
                             compression="gzip")

######################################################################

class LightCurve:
    """
    Light curve base class.  A light curve has a defined
    start and end time.  For any t between start and end,
    lc(t) returns a flux value for the light curve at
    time t. Subclasses of LightCurve implement particular
    functional forms for the light curve.

    Constructor Parameters
    ----------------------
    t_start, t_end : float
      time endpoints of transient

    """

    def __init__(self, t_start, t_end):

        self._t_start = t_start
        self._t_end   = t_end

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_end(self):
        return self._t_end

    def __call__(self, t):
        raise RuntimeError("base class LightCurve has no function defn")

class LightCurveConstant(LightCurve):
    """
    Constant light curve. The total fluence of the curve is equal to
    the scale parameter.

    Constructor Parameters
    ----------------------
    t_start, t_end : float
      time endpoints of transient
    scale : float, optional
      scale factor for total fluence (default 1)

    """

    def __init__(self, t_start, t_end, scale=1.):

        super().__init__(t_start, t_end)
        self._scale = scale / (t_end - t_start)

    def __call__(self, t):
        return self._scale

class LightCurveGaussian(LightCurve):
    """
    Gaussian light curve.  The curve is symmetric between start and
    end time and includes the portion of the Gaussian out to a
    specified number of std deviations from the mean.

    The total fluence of the curve is equal to the scale
    parameter.

    Constructor Parameters
    ----------------------
    t_start, t_end : float
      time endpoints of transient
    n_sds : float, optional
      number of std deviations of normal that fall between
      t_start and t_end (default: 2, or about 95% of
      probability mass)
    scale : float, optional
      scale factor for total fluence (default 1)

    """

    def __init__(self, t_start, t_end,
                 n_sds=2., scale=1.):

        from scipy.stats import norm

        super().__init__(t_start, t_end)

        self.mu = 0.5 * (t_start + t_end)
        self.sd = (t_end - self.mu)/n_sds

        norm = norm.cdf(n_sds) - norm.cdf(-n_sds)
        self._scale = scale / norm

    def __call__(self, t):

        from scipy.stats import norm

        return self._scale * norm.pdf((t - self.mu)/self.sd)

######################################################################

def rand_start_time(source, transient_len, orientations, pad=600.):
    """Choose a random *valid* start time for a transient.  The transient
    is assumed to come from a given source direction and last for
    transient_len seconds.  We select a random start time within the
    range of times for which we have spacecraft orientation data,
    subject to the following restrictions:
      - there is no SAA passage during or within +- 'pad' seconds of the
        transient
      - the source direction is not occluded by earth during
        the transient

    Parameters
    ----------
    source : SkyCoord
      source direction
    transient_len : float
      length of transient
    orientations : SpacecraftFile
      orientation history of spacecraft
    pad : float, optional
      length of required live time in secs before and after transient;
      default = 10 mins

    Returns
    -------
    transient start time in secs (float)
    """

    times = orientations.get_time().to_value("unix")
    history_start = times[0]
    history_len   = times[-1] - times[0]

    while True:

        # choose random start time s.t. transient with padding fits in
        # orientation history
        t_s = history_start + pad + \
            np.random.rand() * (history_len - transient_len - 2 * pad - 0.01)

        # extract the proposed interval from the SpacecraftFile
        ori = orientations.source_interval(Time(t_s - pad,
                                                format="unix"),
                                           Time(t_s + transient_len + pad,
                                                format="unix"))

        # determine the time bins corresponding to the ends of the
        # padded interval and the ends of the transient
        times = ori.get_time().to_value("unix")
        b_s_p, b_s, b_e, b_e_p = np.searchsorted(times,
                                                 [t_s - pad,
                                                  t_s,
                                                  t_s + transient_len,
                                                  t_s + transient_len + pad],
                                                 side='left')
        # get time bins that are occluded by Earth
        is_occluded = ori._get_earth_occ(source.cartesian.xyz)[:-1]

        if np.all(~is_occluded[b_s:b_e]) and np.all(ori.livetime[b_s_p:b_e_p] > 0):
            break

    return t_s

def get_spectrum():

    from astromodels import Band_grbm

    # Use a Band spectrum with Epeak 490 keV, alpha -0.5, beta -2.35

    spectrum = Band_grbm() # matches MegaLib definition
    spectrum.alpha.value = -0.5
    spectrum.beta.value = -2.35
    spectrum.xc.value = 490 # keV

    spectrum.K.value = 1
    spectrum.K.unit = 1/(u.cm**2 * u.s * u.keV)
    spectrum.piv.value = 100
    spectrum.piv.unit = u.keV

    return spectrum

def write_transient_params(output_path, source_loc, light_curve, spectrum):

    with open(output_path, "wt") as outfile:

        lon = source_loc.b.deg
        lat = source_loc.l.deg
        print(f"location {lon} {lat}",
              file = outfile)

        alpha = spectrum.alpha.value
        beta  = spectrum.beta.value
        xc    = spectrum.xc.value
        print(f"spectrum Band 100 10000 {alpha} {beta} {xc} 0",
              file = outfile)

        ts = light_curve.t_start
        te = light_curve.t_end
        print(f"time {ts} {te}",
              file = outfile)

def extract_bg(group, bg_time, tstart, tend, outfile):

    # first index >= tstart
    istart = np.searchsorted(bg_time, tstart, side='left')

    # first index > tend
    iend = np.searchsorted(bg_time, tend, side='right')

    bg_count = iend - istart
    bg_rate  = bg_count / (tend - tstart)

    with h5.File(outfile, "w") as f:

        for dset in group.keys():
            data = group[dset][istart:iend]
            f.create_dataset(dset, data=data, compression="gzip")

    return bg_rate

#####################################################################

data_dir = Path("/project/cassini/cosidata/dc3")
output_dir = Path("/project/cassini/cosi-mapping")

orientation_path = data_dir / "orientation.fits"

response_path = data_dir / "response.h5"

output_path = output_dir / "transients"
output_path.mkdir(parents=True, exist_ok=True)

print("Reading orientations...")
orientations = SpacecraftFile.open(orientation_path)

print("Reading response...")
response = FullDetectorResponse.open(response_path)

print("Reading background times...")
bg = h5.File(data_dir / "bg" / "combined_bg.hdf5")
bg_time = np.array(bg['time'])

# grid for source directions
src_grid = HealpixBase(nside=16, coordsys="galactic")

# define transient spectrum
spectrum = get_spectrum()

# ARGUMENTS: <number of transients> <number of source events per transient> <length of transient>
n_transients = int(sys.argv[1])
nSourceSamples = int(sys.argv[2])
transient_len = float(sys.argv[3]) #seconds

tlstr = str(transient_len).replace(".","-")
output_prefix = f"sim_{nSourceSamples}_{tlstr}"

np.random.seed(1957)

for i in range(n_transients):

    print(f"Generating Transient {i}")

    # pick a random source pixel and use its center as source dir
    src = src_grid.pix2skycoord(np.random.randint(src_grid.npix))

    # pick a random start time s.t. source is live and not occluded
    t_s = rand_start_time(src, transient_len, orientations)

    # define light curve function
    light_curve = LightCurveGaussian(t_s, t_s + transient_len, n_sds=2)

    source_events = get_source_samples(orientations, response,
                                       light_curve, spectrum,
                                       src.cartesian.xyz.value,
                                       nSourceSamples)

    write_source_events(source_events,
                        output_path / f"{output_prefix}_{i}_source.h5")

    extract_bg(bg, bg_time, t_s, t_s + transient_len,
               output_path / f"{output_prefix}_{i}_bkg.h5")

    prior_window = 10*60 # seconds
    extract_bg(bg, bg_time, t_s - prior_window, t_s,
               output_path / f"{output_prefix}_{i}_bkg_prior.h5")

    write_transient_params(output_path / f"{output_prefix}_{i}_params.txt",
                           src, light_curve, spectrum)
