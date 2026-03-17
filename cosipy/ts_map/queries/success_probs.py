import numpy as np

class ECDF:
    """
    Empirical CDF from an array-like of values.
    """

    def __init__(self, values):
        self.values = np.sort(values)
        self.scale = 0. if len(self.values) == 0 else 1./len(self.values)

    def __call__(self, v):
        """
        Return fraction of input values <= v.  If values is empty,
        return 0.
        """
        i = np.searchsorted(self.values, v, side="right")
        return i * self.scale

class SuccessProbs:
    """
    Success probability estimator for mapping + searching

    THEORY OF OPERATION
    -------------------
    This class takes in a CSV file that reports, for each of a large number
    of maps,

      - the numbers of source and background events used to create the map
      - the cost (time) to locate the transient using the map
      - whether the transient was *ever* detected (if not, the search failed)

    It bins the maps in two dimensions: first, by the *total* number
    of events 'e' in the map, and within that, by the number of
    *background* events 'b' in the map. The 'b' bin edges for each
    'e' bin may differ, though the total number of 'b' bins per 'e'
    bin is fixed.

    Within each 2D bin, we compute an empirical CDF of the mapping
    time costs, treating failed searches as taking effectively
    infinite time.

    Once built, we can query the object to get an estimate of the
    probability of success given the total number of events e, the
    number of *background* events b, and the time remaining for the
    partner to plan and execute a search for the transient given the
    map.

    """
    def __init__(self, csv_path, nbins_e, nbins_b, min_bin_count):
        """
        Parameters
        ----------
        csv_path : string or Path object
          path to CSV file containing mapping results
        nbins_e : int
          number of evenly spaced bins to divide the 'e' dimension
        nbins_b : int
          number of evenly spaced bins to divide the 'b' dimension for each e
        min_bin_count : int
          smallest number of maps for which a bin is considered to have
          a trustworthy ECDF.
        """

        import pandas as pd

        df = pd.read_csv(csv_path)

        FAILURE_COST=1e+9 # too high to ever succeed
        df["detection_cost"] = df["detection_cost"].where(df["detected"],
                                                          FAILURE_COST)

        n_src = df["n_src"].values.astype(int)
        n_bkg = df["n_bkg"].values.astype(int)
        n_total = n_src + n_bkg

        cost = df["detection_cost"].values.astype(float)

        # Divide map costs into bins, and compute an ECDF for the
        # costs in each bin.
        #
        # Bins are assumed to be half-open, including low endpt but
        # not high endpt. Bin edges are integer-valued and so are
        # not necessarily exactly equally spaced.

        self.e_edges = np.linspace(n_total.min(), n_total.max() + 1,
                                   nbins_e + 1)
        self.e_edges = np.ceil(self.e_edges).astype(int)

        # storage for 'b' edges and ECDFs
        self.b_edges = np.empty((nbins_e, nbins_b + 1), dtype=int)
        self.ecdfs   = np.empty((nbins_e, nbins_b), dtype=object)

        for i in range(nbins_e):
            # select maps in this 'e' bin
            emask = ((n_total >= self.e_edges[i]) &
                     (n_total <  self.e_edges[i+1]))
            n_bkg_i = n_bkg[emask]
            cost_i  = cost[emask]

            # compute 'b' edges for this bin
            b_edges = np.linspace(n_bkg_i.min(), n_bkg_i.max() + 1,
                                  nbins_b + 1)
            self.b_edges[i] = np.ceil(b_edges).astype(int)

            # collect maps for each 'b' bin in descending order by 'b',
            # to support filtering out of too-small bins
            for j in reversed(range(nbins_b)):

                # select maps in this (e,b) bin
                bmask = ((n_bkg_i >= self.b_edges[i,j]) &
                         (n_bkg_i <  self.b_edges[i,j+1]))
                cost_ij = cost_i[bmask]

                if len(cost_ij) < min_bin_count:
                    # too little data for a valid ECDF; use a
                    # pessimistic estimate of the ECDF instead
                    if j == nbins_b - 1:
                        # ECDF always returns 0
                        self.ecdfs[i,j] = ECDF(np.array([]))
                    else:
                        # use next higher (more pessimistic) 'b' bin
                        self.ecdfs[i,j] = self.ecdfs[i,j+1]
                else:
                    self.ecdfs[i,j] = ECDF(cost_ij)

    def _get_e_bin(self, n_total):
        """
        Return the index of the 'e' bin containing the 'e' value n_total.
          - If n_total is less than the least 'e' edge, return -1.
          - If n_total is more than the greatest 'e' edge, return the
            greatest 'e' bin.
        """

        # more total events than we've ever seen
        n_total = np.minimum(n_total, self.e_edges[-1] - 1)

        return np.searchsorted(self.e_edges, n_total, side="right") - 1

    def get_b_edges(self, n_total):
        """
        Return the set of 'b' edges for the 'e' bin containing the
        'e' value n_total.
          - If n_total is less than the least 'e' edge, return None.
          - If n_total is more than the greatest 'e' edge, return the
            'b' edges for the greatest 'e' bin.
        """

        i = self._get_e_bin(n_total)
        if i < 0: # fewer total events than we've ever seen
            return None
        else:
            return self.b_edges[i]

    def __call__(self, n_total, n_bkg, deadline):
        """
        Return the estimated success probability for a map built from
        n_total events, of which n_bkg are background, assuming that
        there are 'deadline' seconds remaining after mapping to search
        for the transient.

        """

        i = self._get_e_bin(n_total)
        if i < 0:  # fewer total events than we've ever seen
            return 0.

        b_edges = self.b_edges[i]

        if n_bkg >= b_edges[-1]: # more background events than we've ever seen
            return 0.

        # fewer background events than we've ever seen
        n_bkg = np.maximum(n_bkg, b_edges[0])

        # get the 'b' bin in which n_bkg falls
        j = np.searchsorted(b_edges, n_bkg, side="right") - 1

        return self.ecdfs[i,j](deadline)
