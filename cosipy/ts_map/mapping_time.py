import numpy as np

def choose_mapping_time(events_per_sec, bkg_rate):
    """
    Events_per_sec: total number of events observed each second

    RETURNS: where to cut data, time until decision was made

    """

    from scipy.stats import poisson

    max_wait = 3 # time after last significant second to wait
    significance = 0.95 # threshold for significant excess


    last_significant = -1

    # process events for each second
    for t, e in enumerate(events_per_sec):

        bpois = poisson.cdf(e, mu=bkg_rate)

        if (bpois >= significance): # current bin is above bg
            last_significant = t
        elif t - last_significant >= max_wait:
            return (last_significant+1, t+1)

    # did not terminate -- return end of last significant bin
    return (last_significant+1, len(events_per_sec))


def trim_events(events, t_start, bkg_rate, events_per_sec):

    (length, mapping_time) = choose_mapping_time(events_per_sec, bkg_rate)
    t_end = t_start + length

    e_end = np.searchsorted(events["time"].value, t_end, side='right')

    events["time"]   = events["time"][:e_end]
    events["Em"]     = events["Em"][:e_end]
    events["Phi"]    = events["Phi"][:e_end]
    events["PsiChi"] = events["PsiChi"][:, :e_end]

    return t_end, mapping_time
