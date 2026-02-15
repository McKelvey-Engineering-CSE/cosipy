from pathlib import Path
import re

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
                                        "time", "energy", "eta",
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
        time = df["time"].values + 1580000000 # set burst time to ~2020 in Unix
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
                "time"    : time,
                "Em"      : Em,
                "Phi"     : phi,
                "Chi"     : chi,
                "Psi"     : psi,
            })
        self.df.set_index("eventid", inplace=True)


    def sample_events(self, n_raw_events):

        # First, determine how many of n_raw_events random samples
        # from the range [0, self.id_range) *without replacement*
        # would fall among the events actually present in the file.

        n_events_present = len(self.event_ids)
        n_events = np.random.hypergeometric(n_events_present,
                                            self.id_range - n_events_present,
                                            n_raw_events)

        # Next, select that many events from the list of events in the
        # file without replacement

        event_ids = np.random.choice(self.event_ids,
                                     size=n_events,
                                     replace=False)

        # Finally, return all rings corresponding to the selected events.

        return self.df.loc[self.df.index.isin(event_ids)]


def write_sample(df, output_file):
    # write unbinned event format expected by cosipy
    with h5.File(output_file, "w") as f:
        f.create_dataset("time", data=df.time.values)
        f.create_dataset("Em", data=df.Em.values)
        f.create_dataset("Phi", data=df.Phi.values)
        f.create_dataset("Psi", data=df.Psi.values)
        f.create_dataset("Chi", data=df.Chi.values)


np.random.seed(1957)

ring_dir = Path("/project/starkiller/scratch0/nn_geant_adapt_10m/0_seed")
data_dir = Path("/project/cassini/adapt_grbs")
bg_dir   = Path("/project/cassini/adapt_grbs/source/bg")
#bg_dir   = Path("/project/starkiller/scratch0/geant_adapt_background/500_seed")

output_dir = data_dir / "adapt_transients"
output_dir.mkdir(parents=True, exist_ok=True)

bg_time = 1.     # seconds of bg time
src_mean = 44600 # mean number of events expected -- 1 MeV/cm^2 fluence

NBursts_per_src_dir = 100 # generate this many bursts per source direction

# names of all background components, number of generated events, and
# number of events to sample for a 1 second burst
bg_components = {
    "gamma"   : (300000000, 357878),
    "e-"      : (300000000, 28548),
    "proton"  : (300000000, 4673),
    "neutron" : (300000000, 57250)
}

# number of events expected from each bg component
bg_means = np.array([bg_components[c][1] for c in bg_components]) * bg_time

# load data sets for each bg component
bg_data = {}
for c in bg_components:
    #bg_file = bg_dir / f"{c}_{bg_components[c][0]}" / "circles_p0_a0.txt"
    bg_file = bg_dir / c / "circles.parquet"

    print(f"Reading {bg_file}")
    bg_data[c] = ParticleSet(bg_file, bg_components[c][0])


# enumerate all the source directions in the subdir
rexp = re.compile(r"p([0-9]+)_a([0-9]+)")

for src_dir in ring_dir.glob("*"):
    name = src_dir.name
    print(name)

    m = re.match(rexp, name)
    alt = 90 - int(m.group(1))
    az  = int(m.group(2))

    with open(output_dir / f"adapt_{name}_params.txt", "w") as f:
        print(f"location {alt} {az}", file=f)
        print("spectrum Band 30 30000 -0.5 -2.35 490", file=f)

    ring_file = src_dir / f"circles_{name}.txt"

    # events (unique eventids) generated per source file
    n_generated_events = 10000000
    src_ps = ParticleSet(ring_file, n_generated_events)

    for i in range(NBursts_per_src_dir):

        # determine how
        n_src_events = np.random.poisson(src_mean)

        src_ds = src_ps.sample_events(n_src_events)
        write_sample(src_ds, output_dir / f"adapt_{name}_{i}_source.h5")

        # determine how many event IDs to sample from each bg type,
        # allowing for variation about the means
        n_bg_events = np.random.poisson(bg_means)

        bg_all_ds = [
            bg_ps.sample_events(n_ev)
            for bg_ps, n_ev
            in zip(bg_data.values(), n_bg_events)
        ]

        bg_ds_combined = pd.concat(bg_all_ds)
        write_sample(bg_ds_combined,
                     output_dir / f"adapt_{name}_{i}_background.h5")
