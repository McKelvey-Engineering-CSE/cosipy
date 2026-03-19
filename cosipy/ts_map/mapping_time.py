import numpy as np

from scipy.stats import poisson

from queries.mapping_time_query import MapTimeQueryEngine
from queries.success_probs      import SuccessProbs

time_quantile = 0.95   # quantile of mapping time used to estimate utility
success_conf  = 0.95   # quantile for success prob confidence lower bound

mapping_time_data = "queries/emsoft.csv"
success_prob_data = "queries/search_result_5.36x4.5_tiling.csv"

# This class is from Daisy
mapping_time = MapTimeQueryEngine(csv_path = mapping_time_data,
                                  quantiles = [time_quantile],
                                  n_bins_per_axis = 20,
                                  min_bin_count = 10)

success_prob = SuccessProbs(csv_path = success_prob_data,
                            nbins_e = 20,
                            nbins_b = 20,
                            min_bin_count = 20,
                            conf = success_conf)

def choose_mapping_start_time(events_per_time_step, delta_t, bkg_rate,
                              deadline_overall):
    """
    Determine the *last* time at which we begin to compute a
    likelihood map given observations of the total number of events
    arriving in each 1-second period after some externally
    determined starting time.

    At the end of each 1-second time period, we estimate the utility
    of mapping at that time, defined as the probability that the
    partner instrument will successfully discover the transient by an
    externally imposed deadline. If the utility is higher than at any
    previous time, we restart mapping.

    Rather than actually stopping and restarting mapping, this
    procedure does the utility calculation for *all* periods, up to
    the number of seconds of event counts provided, all at once and
    determines the last time at which we would restart mapping.

    Empiricially, this procedure costs only tens of ms per second of
    observation, so adds a negligible amount to the actual waiting,
    mapping, and search cost.

    Parameters
    ----------
    events_per_time_step : array of int
      number of new events (source + bkg) arriving each time step after
      the start of the burst, for some sufficiently long period.
    delta_t : float
      length of one time step in seconds
    bkg_rate : float
      estimated mean number of background events arriving per second
    deadline_overall : float
      time by which we must find the transient in order to succeed

    Returns
    -------
     t_end : float
       time in seconds after start beyond which we should not consider
       events as input to mapping. <= mapping_time
     mapping_time : float
       time in seconds after start at which we last start to produce a
       map; any previous starts are assumed to be suppressed

    """

    # compute total events seen after each second
    total_events = np.cumsum(events_per_time_step)

    Uavg = np.zeros(len(total_events))

    # compute utilities at end of each second
    for j in range(1, len(total_events) + 1): # end time
        t_wait = j * delta_t
        e = total_events[j-1]

        b_edges = success_prob.get_b_edges(e)
        if b_edges is None: # too few total events
            Uavg[j-1] = 0.
        else:
            # Because binning by total events combines several 'e'
            # values in one bin, the highest 'b' value that can occur
            # in an 'e' bin can can exceed some actual total event
            # count 'e' that falls into that bin.  Make sure we never
            # consider our background event count never exceeds the actual
            # value of 'e' we are using while computing utility.
            b_edges = np.where(b_edges > e, e + 1, b_edges)

            # Total Poisson weight for all the 'b' values in each
            # half-open 'b' bin, given the total number of background
            # events expected after t seconds.
            #
            # Weights are normalized to sum to 1 for b in the range
            # [0..e].

            w_pois = \
                np.diff(poisson.cdf(b_edges - 1, bkg_rate * t_wait)) / \
                poisson.cdf(e, bkg_rate * t_wait)

            # add utility contributions of each 'b' bin

            nbins_b = len(b_edges) - 1
            for i in range(nbins_b):

                if w_pois[i] < 1e-10: # skip bins with negligible weight
                    continue

                bh = b_edges[i+1] - 1

                # Use pessimistic mapping time estimate -- using as
                # few source events as possible given 'e' should
                # maximize mapping time.
                t_mapping = mapping_time.query(n_bkg = bh,
                                               n_src = e - bh,
                                               quantile = time_quantile)

                # 'b' binning and pessimistic estimate of t_mapping
                # ensure that success_prob returns same utility for
                # every 'b' in this bin.  WLOG we use the largest 'b'
                # value in the bin.
                sp = success_prob(deadline = deadline_overall - t_wait - t_mapping,
                                  n_total = e, n_bkg = bh)

                # Because we get only 100qth %ile mapping time, we
                # must assume that with probability 1 - q, that time
                # is arbitarily large, exceeding our overall deadline
                # and leading to failure.
                utility = time_quantile * success_conf * sp

                # if utility goes to 0 for some 'b' bin, it will be 0
                # for all higher 'b' bins. Don't add 0's to the
                # average utility for this time
                if utility == 0:
                    break

                Uavg[j-1] += utility * w_pois[i]

            # if *zero* utility at current time step, assume utility
            # will be zero for all later time steps becaue the input
            # WHP has too much background for us to predict utility > 0.
            if Uavg[j-1] == 0.:
                break

    # pick end of time period with greatest utility
    t_max = (np.argmax(Uavg) + 1) * delta_t

    # This method never "backs up" to an earlier end time for the
    # transient, so the end time is always the same as the time at
    # which we start mapping.  We could add a guess at this method's
    # compute time to t_max, but it's so small (<< 1 s) as to be
    # negligible.
    return float(t_max), float(t_max)


def trim_events(events, t_start, bkg_rate, deadline_overall,
                delta_t = 1):

    """
    Trim a set of Compton events with associated time stamps to
    just the range we want to use for mapping.  The decision
    is made based on a method choose_mapping_start_time().

    Parameters
    ----------
    events : dict
      Event arrival times and CDS data, sorted by arrival time
    t_start : float
      Time at which the instrument triggers
    bkg_rate : float
      Estimated mean arrival rate (events/sec) for background Compton
      rings
    deadline_overall : float
      Time by which we must find the transient to succeed
    delta_t : float, optional
      Length of time steps into which we divide events for end time
      determination

    Returns
    -------
     t_end : float
       time in seconds after start beyond which we should not consider
       events as input to mapping. <= mapping_time
     mapping_time : float
       time in seconds after start at which we last start to produce a
       map; any previous starts are assumed to be suppressed

    The input event set is trimmed to the range [t_start, t_end] in place.

    """

    # compute the total number of events arriving each second
    edges = \
        t_start + np.arange(np.ceil(events["time"][-1].value) - t_start + 1,
                            step=delta_t)
    events_per_time_step, _ = np.histogram(events["time"].value, edges)

    # determine burst length
    length, mapping_time = choose_mapping_start_time(events_per_time_step,
                                                     delta_t,
                                                     bkg_rate,
                                                     deadline_overall)
    t_end = t_start + length

    # keep only events occurring before t_end
    e_end = np.searchsorted(events["time"].value, t_end, side='right')

    events["time"]   = events["time"][:e_end]
    events["Em"]     = events["Em"][:e_end]
    events["Phi"]    = events["Phi"][:e_end]
    events["PsiChi"] = events["PsiChi"][:, :e_end]

    return t_end, mapping_time
