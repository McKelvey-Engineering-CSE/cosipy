#
# USAGE:
# gen_adapt_transients_dense <bkg_length> <ntrans_per_source_dir> <output>
#
#  bkg_length : length of background to generate in seconds
#  ntrans_per_source_dir : number of bursts to generate per source direction
#  output : directory to write output files (created if not present)

from pathlib import Path
import re
import sys

import numpy as np

import pandas as pd

import h5py as h5

class ParticleSet():

    def __init__(self, filename, id_range):

        self.id_range = id_range

        if filename.suffix == ".parquet":
            df = pd.read_parquet(filename)
        else:
            df = pd.read_csv(filename, sep=' ', index_col=False,
                             usecols = ["eventid", "below90", "same_layer",
                                        "energy", "eta",
                                        "c0", "c1", "c2"])

        # apply pipeline filters between reconstruction and localization
        df = df.loc[df["below90"] == 0]
        df = df.loc[df["same_layer"] == 0]
        df = df.loc[np.abs(df["eta"]) <= 1.]

        # remove events outside the energy range of the response
        df = df[(df["energy"] * 511 >= 100.) & (df["energy"] * 511 <= 10000.)]

        # save all the unique event IDs
        self.event_ids = np.unique(df["eventid"])

        # extract relevant columns and convert to desired format
        Em   = df["energy"].values * 511 # keV
        phi  = np.acos(df["eta"].values) # rad
        phi[phi == np.pi] -= 1e-12 # prevent out-of-range value for Phi axis

        x = df["c0"].values
        y = df["c1"].values
        z = df["c2"].values

        chi = np.atan2(y, x) # longitude (rad)
        psi = np.acos(z)     # co-latitude (rad)

        self.df = pd.DataFrame(
            {
                "eventid" : df["eventid"],
                "Em"      : Em,
                "Phi"     : phi,
                "Chi"     : chi,
                "Psi"     : psi,
            })
        self.df.set_index("eventid", inplace=True)

    def observed_event_fraction(self):
        # number of observed events per incident event
        return len(self.event_ids) / self.id_range

    def avg_rings_per_event(self):
        # mean number of rings per observed event
        return len(self.df)/len(self.event_ids)

    def estimate_sample_size(self, n_raw_events):

        # Determine how many of n_raw_events random samples from the
        # range [0, self.id_range) *without replacement* would fall
        # among the events actually present in the file.

        n_events_present = len(self.event_ids)
        n_events = np.random.hypergeometric(n_events_present,
                                            self.id_range - n_events_present,
                                            n_raw_events)

        return n_events

    def sample_events(self, n_raw_events):

        n_events = self.estimate_sample_size(n_raw_events)

        # Next, select that many events from the list of events in the
        # file without replacement

        event_ids = np.random.choice(self.event_ids,
                                     size=n_events,
                                     replace=False)

        # Sort event IDs so we can quickly locate the index of each
        # ring's event in this array later

        event_ids = np.sort(event_ids)

        # Finally, return the list of selected event IDs, along with
        # all rings corresponding to these selected events.
        return event_ids, self.df.loc[self.df.index.isin(event_ids)]

def write_sample_src(df, output_file):

    # write unbinned event format expected by cosipy
    with h5.File(output_file, "w") as f:
        f.create_dataset("time", data=df.times.values)
        f.create_dataset("Em", data=df.Em.values)
        f.create_dataset("Phi", data=df.Phi.values)
        f.create_dataset("Psi", data=df.Psi.values)
        f.create_dataset("Chi", data=df.Chi.values)

def write_sample_bg(df, output_file, prior):

    # write unbinned event format expected by cosipy
    with h5.File(output_file, "w") as f:
        f.create_dataset("time", data=df.times.values)
        f.create_dataset("Em", data=df.Em.values)
        f.create_dataset("Phi", data=df.Phi.values)
        f.create_dataset("Psi", data=df.Psi.values)
        f.create_dataset("Chi", data=df.Chi.values)

        # record prior estimate of background rate
        f.attrs["bg_prior"] = prior


ring_dir = Path("/project/starkiller/scratch0/nn_geant_adapt_10m/0_seed")
data_dir = Path("/project/cassini/adapt_grbs")
bg_dir   = Path("/project/cassini/adapt_grbs/source/bg")
#bg_dir   = Path("/project/starkiller/scratch0/geant_adapt_background/500_seed")
bg_length   = float(sys.argv[1]) # length of bg to generate in seconds
n_bursts_per_src_dir = int(sys.argv[2]) # number of bursts per source direction
output_dir  = Path(sys.argv[3])  # where to write output bursts

output_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(1957)

# arbitrary start time for burst
start_offset = 1580000000 # ~2020 in UNIX time base

# range for source length and fluence (will sample uniformly)
src_length_range  = (0.25, 5.0)
src_fluence_range = (0.5,  5.0)

# names of all background components, number of generated events, and
# number of events to sample for a 1-second burst
bg_components = {
    "gamma"   : (300000000, 357878),
    "e-"      : (300000000, 28548),
    "proton"  : (300000000, 4673),
    "neutron" : (300000000, 57250)
}

# load data sets for each bg component
bg_data = {}
for c in bg_components:
    bg_file = bg_dir / c / "circles.parquet"
    print(f"Reading {bg_file}")

    bg_data[c] = ParticleSet(bg_file, bg_components[c][0])

######################################################################

# number of events expected from each bg component
bg_means = np.array([
    bg_components[c][1]
    for c in bg_components
]) * bg_length

# number of *observed* events expected from each bg component in 10
# minutes
bg_prior_time = 10*600.
bg_prior_observed_ring_means = np.array([
    bg_components[c][1] * bg_ps.observed_event_fraction() * bg_ps.avg_rings_per_event()
    for c, bg_ps in bg_data.items()
]) * bg_prior_time

# enumerate all the source directions in the subdir
rexp = re.compile(r"p([0-9-]+)_a([0-9-]+)")
src_dirs = list(ring_dir.glob("*"))
src_dirs.sort()

for src_dir in src_dirs:
    src_name = src_dir.name
    out_name = src_name

    print(out_name)

    m = re.match(rexp, src_name)
    alt = 90. - float(m.group(1).replace("-","."))
    az  = float(m.group(2).replace("-","."))

    ring_file = src_dir / f"circles_{src_name}.txt"

    # events (unique eventids) generated per source file
    n_generated_events = 10000000
    src_ps = ParticleSet(ring_file, n_generated_events)

    for i in range(n_bursts_per_src_dir):

        # randomize source fluence and length
        src_fluence = np.random.rand() * \
            (src_fluence_range[1] - src_fluence_range[0]) + \
            src_fluence_range[0]

        src_length = np.random.rand() * \
            (src_length_range[1] - src_length_range[0]) + \
            src_length_range[0]

        # write *true* event parameters
        with open(output_dir / f"adapt_{out_name}_{i}_params.txt", "w") as f:
            print(f"location {alt} {az}", file=f)
            print("spectrum Band 30 30000 -0.5 -2.35 490", file=f)
            print(f"light_curve {start_offset} {src_length} {src_fluence} normal_centered 2", file=f)

        # expected number of events -- base value is for 1 MeV/cm^2 fluence
        src_mean = 44600 * src_fluence

        # determine how many source events this burst contains
        n_src_events = np.random.poisson(src_mean)

        src_event_ids, src_ds = src_ps.sample_events(n_src_events)

        # generate times for each source event according to Gaussian
        # light curve.  mean time is length/2; start at 0, end at
        # length have brightness +- 2 sdevs down from mean
        times = np.random.normal(loc=src_length/2, scale=(src_length/2)/2,
                                 size=2*len(src_ds))
        times = times[(times >= 0.) & (times <= src_length)]
        assert len(times) >= len(src_event_ids)
        times = times[:len(src_event_ids)] + start_offset

        # for each event ID, set all rings with that ID to
        # corresponding time
        indices = np.searchsorted(src_event_ids, src_ds.index.values)
        src_ds["times"] = times[indices]
        src_ds.sort_values(by="times")

        write_sample_src(src_ds,
                         output_dir / f"adapt_{out_name}_{i}_source.h5")

        # compute the rate (rings/sec) we'd estimate for the
        # background from bg_prior_time seconds' worth of observations
        n_prior_bg_rings = np.random.poisson(bg_prior_observed_ring_means)
        bg_prior_rate = np.sum(n_prior_bg_rings) / bg_prior_time

        # determine how many event IDs to sample from each bg type,
        # allowing for variation about the means
        n_bg_events = np.random.poisson(bg_means)

        bg_all_ds = []
        for bg_ps, n_ev in zip(bg_data.values(), n_bg_events):
            # generate times uniformly within bg length
            bg_event_ids, bg_ds = bg_ps.sample_events(n_ev)
            times = np.random.rand(len(bg_event_ids)) * bg_length + start_offset
            # for each event ID, set all rings with that ID to
            # corresponding time
            indices = np.searchsorted(bg_event_ids, bg_ds.index.values)
            bg_ds["times"] = times[indices]

            bg_all_ds.append(bg_ds)

        bg_ds_combined = pd.concat(bg_all_ds)
        bg_ds_combined.sort_values(by="times")

        write_sample_bg(bg_ds_combined,
                        output_dir / f"adapt_{out_name}_{i}_background.h5",
                        prior = bg_prior_rate)
