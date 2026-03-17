"""
Mapping time query engine.
Usage
-----
1. Fits parametric models to quantile levels of mapping_time
    engine = MapTimeQueryEngine("emsoft.csv")

2. Query as needed after
    t = engine.query(n_src=400, n_bkg=900, quantile=0.99)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def _qkey(q):
    return f"q{round(q * 100000):06d}"

# models
def power_src_bkg(X, a, b, c, d):
    ns = np.maximum(X[0], 1.0); nb = np.maximum(X[1], 1.0)
    return a * (ns ** (-b)) * (nb ** c) + d

def exp_ratio_bkg(X, a, b, c, d):
    nb = np.maximum(X[1], 1.0)
    return a * np.exp(-b * (X[0] / nb)) * (nb ** c) + d

def mm_ratio_bkg(X, a, b, c, d):
    nb = np.maximum(X[1], 1.0)
    return a * (nb ** c) / (1.0 + b * (X[0] / nb)) + d

def stretched_exp_ratio_bkg(X, a, b, k, c, d):
    nb = np.maximum(X[1], 1.0)
    t  = np.power(np.maximum(b * X[0] / nb, 0.0), k)
    return a * np.exp(-t) * (nb ** c) + d

def stretched_exp_src_times_bkg(X, a, b, k, c, d):
    ns = np.maximum(X[0], 1.0); nb = np.maximum(X[1], 1.0)
    t  = np.power(np.maximum(b * ns, 0.0), k)
    return a * np.exp(-t) * (nb ** c) + d

def logistic_ratio_bkg(X, a, b, r0, k, c, d):
    nb  = np.maximum(X[1], 1.0)
    sig = 1.0 / (1.0 + np.exp(np.clip(k * (X[0] / nb - r0), -60.0, 60.0)))
    return a * (nb ** c) * sig + d

def softplus_gap(X, a, alpha, tau, c, d):
    nb = np.maximum(X[1], 1.0)
    z  = np.clip((X[1] - alpha * X[0]) / np.maximum(tau, 1e-6), -60.0, 60.0)
    return a * np.log1p(np.exp(z)) * (nb ** c) + d

def loglink_interaction(X, a, b, c, e, d):
    Ls = np.log1p(np.maximum(X[0], 1.0))
    Lb = np.log1p(np.maximum(X[1], 1.0))
    return np.exp(np.clip(a + b * Lb - c * Ls + e * Lb * Ls, -60.0, 60.0)) + d


MODELS = {
    "power_src_bkg":               (power_src_bkg,               [1e4, 0.5, 0.5, 10],               "a/src^b * bkg^c + d"),
    "exp_ratio_bkg":               (exp_ratio_bkg,               [100, 1, 0.1, 10],                 "a*exp(-b*r)*bkg^c + d"),
    "mm_ratio_bkg":                (mm_ratio_bkg,                [500.0, 1.0, 0.5, 10.0],           "a*bkg^c/(1+b*r)+d"),
    "stretched_exp_ratio_bkg":     (stretched_exp_ratio_bkg,     [500.0, 1.0, 0.7, 0.5, 10.0],      "a*exp(-(b*r)^k)*bkg^c+d"),
    "stretched_exp_src_times_bkg": (stretched_exp_src_times_bkg, [500.0, 1e-2, 0.7, 0.5, 10.0],     "a*exp(-(b*src)^k)*bkg^c+d"),
    "logistic_ratio_bkg":          (logistic_ratio_bkg,          [500.0, 1.0, 0.5, 5.0, 0.5, 10.0], "a*bkg^c*sig(k*(r-r0))+d"),
    "softplus_gap":                (softplus_gap,                [50.0, 2.0, 50.0, 0.0, 10.0],      "a*sp((bkg-a*src)/tau)*bkg^c+d"),
    "loglink_interaction":         (loglink_interaction,         [3.0, 0.5, 0.5, 0.0, 0.0],         "exp(a+b*lnB-c*lnS+e*lnB*lnS)+d"),
}


# binning
def _bin_stats_2d(n_src, n_bkg, cost, n_bins_per_axis, quantiles):
    """
    Bin on a 2D equal-width grid and compute per-bin quantiles.

    Returns
    -------
    binned    : dict[_qkey(q)] -> {X, y, counts}
    src_edges, bkg_edges
    """
    src_edges = np.linspace(n_src.min(), n_src.max(), n_bins_per_axis + 1)
    bkg_edges = np.linspace(n_bkg.min(), n_bkg.max(), n_bins_per_axis + 1)
    src_idx   = np.digitize(n_src, src_edges[1:-1])
    bkg_idx   = np.digitize(n_bkg, bkg_edges[1:-1])

    qkeys   = [_qkey(q) for q in quantiles]
    targets = {k: [] for k in qkeys}
    bin_src, bin_bkg, bin_counts = [], [], []

    for si in range(len(src_edges) - 1):
        for bi in range(len(bkg_edges) - 1):
            mask = (src_idx == si) & (bkg_idx == bi)
            if mask.sum() < 2:
                continue
            c = cost[mask]
            bin_src.append(0.5 * (src_edges[si] + src_edges[si + 1]))
            bin_bkg.append(0.5 * (bkg_edges[bi] + bkg_edges[bi + 1]))
            bin_counts.append(mask.sum())
            for q, k in zip(quantiles, qkeys):
                targets[k].append(np.percentile(c, q * 100))

    bin_src    = np.array(bin_src,    dtype=float)
    bin_bkg    = np.array(bin_bkg,    dtype=float)
    bin_counts = np.array(bin_counts, dtype=int)

    binned = {}
    for k in qkeys:
        binned[k] = dict(
            X=(bin_src.copy(), bin_bkg.copy()),
            y=np.array(targets[k], dtype=float),
            counts=bin_counts.copy(),
        )
    return binned, src_edges, bkg_edges

def _fit_models_to_target(X_bin, y_bin, sigma=None):
    results = {}
    for name, (func, p0, formula) in MODELS.items():
        try:
            popt, _ = curve_fit(func, X_bin, y_bin, p0=p0, maxfev=1000000,
                                sigma=sigma, absolute_sigma=False)
            y_pred = func(X_bin, *popt)
            r2     = r2_score(y_bin, y_pred)
            rmse   = float(np.sqrt(np.mean((y_bin - y_pred) ** 2)))
            mae    = float(np.mean(np.abs(y_bin - y_pred)))
            n_p, n_d = len(popt), len(y_bin)
            adj_r2 = 1 - (1 - r2) * (n_d - 1) / max(n_d - n_p - 1, 1)
            results[name] = dict(params=popt, r2=r2, adj_r2=adj_r2,
                                rmse=rmse, mae=mae, func=func, formula=formula)
        except Exception as e:
            print(f"    {name:30s}: FAILED ({e})", file=sys.stderr)
    return {k: v for k, v in results.items() if v is not None}


def _best_model(valid):
    return max(valid.items(), key=lambda kv: kv[1]["r2"])


class MapTimeQueryEngine:

    def __init__(self, csv_path,
                 quantiles=(0.90, 0.99, 0.999),
                 n_bins_per_axis=20,
                 min_bin_count=10):
        """
        Parameters
        ----------
        csv_path        : path to CSV (must have n_src, n_bkg, mapping_time)
        quantiles       : quantile levels to fit
        n_bins_per_axis : 2D grid resolution
        min_bin_count   : bins with fewer samples are dropped before fitting
        """
        self.quantiles = sorted(quantiles)
        self._qkeys    = [_qkey(q) for q in self.quantiles]

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df):,} rows from {csv_path}", file=sys.stderr)
        print(f"  n_src       : [{df['n_src'].min():.0f}, {df['n_src'].max():.0f}]", file=sys.stderr)
        print(f"  n_bkg       : [{df['n_bkg'].min():.0f}, {df['n_bkg'].max():.0f}]", file=sys.stderr)
        print(f"  mapping_time: [{df['mapping_time'].min():.3f}, "
              f"{df['mapping_time'].max():.3f}] s", file=sys.stderr)

        n_src = df["n_src"].values.astype(float)
        n_bkg = df["n_bkg"].values.astype(float)
        cost  = df["mapping_time"].values.astype(float)

        binned, src_edges, bkg_edges = _bin_stats_2d(
            n_src, n_bkg, cost, n_bins_per_axis, self.quantiles)

        #Filter sparse bins
        counts = binned[self._qkeys[0]]["counts"]
        mask   = counts >= min_bin_count
        print(f"  Dropped {(~mask).sum()} bins with <{min_bin_count} samples, "
              f"{mask.sum()} remain", file=sys.stderr)

        for k in self._qkeys:
            b = binned[k]
            binned[k] = dict(
                X=(b["X"][0][mask], b["X"][1][mask]),
                y=b["y"][mask],
                counts=b["counts"][mask],
            )

        #Fit best model per quantile
        self._models = {}
        for q, k in zip(self.quantiles, self._qkeys):
            X_bin  = binned[k]["X"]
            y_bin  = binned[k]["y"]
            counts = binned[k]["counts"]
            sigma  = 1.0 / np.sqrt(np.maximum(counts, 1))

            valid = _fit_models_to_target(X_bin, y_bin, sigma=sigma)
            if not valid:
                print(f"  WARNING: no model converged for q={q}", file=sys.stderr)
                continue

            name, r = _best_model(valid)
            self._models[k] = dict(
                func=r["func"], params=r["params"],
                name=name, r2=r["r2"], formula=r["formula"],
            )
            print(f"  q={q}  best={name:30s}  R2={r['r2']:.4f}", file=sys.stderr)

        # for diagnostics
        self._binned    = binned
        self._src_edges = src_edges
        self._bkg_edges = bkg_edges
        self._n_src_raw = n_src
        self._n_bkg_raw = n_bkg
        self._cost_raw  = cost

    # Internal helpers
    def _eval(self, k, n_src, n_bkg):
        """Evaluate fitted model for key k at a single (n_src, n_bkg) point."""
        m = self._models[k]
        return float(m["func"](
            (np.atleast_1d(float(n_src)), np.atleast_1d(float(n_bkg))),
            *m["params"])[0])

    # query interface
    def query(self, n_src, n_bkg, quantile=0.99):
        """
        Predicted mapping_time at the given quantile level.

        If quantile matches a fitted level exactly, uses that model directly.
        If quantile falls between two fitted levels, LINEARLY interpolates.

        Parameters
        ----------
        n_src    : float
        n_bkg    : float
        quantile : float — does not need to exactly match a fitted level

        Returns
        -------
        t : float  predicted mapping time in seconds
        """
        k = _qkey(quantile)

        # Exact match
        if k in self._models:
            return self._eval(k, n_src, n_bkg)

        # Interpolate between bracketing quantiles
        fitted_qs = sorted(self.quantiles)
        if quantile < fitted_qs[0] or quantile > fitted_qs[-1]:
            raise ValueError(
                f"Quantile {quantile} is outside the fitted range "
                f"[{fitted_qs[0]}, {fitted_qs[-1]}]. "
                f"Rebuild the engine with a wider quantile list.")

        lo = max(q for q in fitted_qs if q <= quantile)
        hi = min(q for q in fitted_qs if q >= quantile)
        t  = (quantile - lo) / (hi - lo)

        y_lo = self._eval(_qkey(lo), n_src, n_bkg)
        y_hi = self._eval(_qkey(hi), n_src, n_bkg)
        return (1 - t) * y_lo + t * y_hi

    def query_all(self, n_src, n_bkg):
        """
        Return predicted mapping time for every fitted quantile.

        Returns
        -------
        dict {quantile: predicted_time}
        """
        return {q: self.query(n_src, n_bkg, quantile=q)
                for q in self.quantiles}

    def batch_query(self, queries, quantile=0.99):
        """
        Query multiple (n_src, n_bkg) pairs at once.

        Parameters
        ----------
        queries : array-like (N, 2) columns [n_src, n_bkg]
                  or DataFrame with those column names

        Returns
        -------
        DataFrame: n_src, n_bkg, predicted_time
        """
        if isinstance(queries, pd.DataFrame):
            rows = queries[["n_src", "n_bkg"]].values
        else:
            rows = np.asarray(queries)

        t = np.array([self.query(ns, nb, quantile=quantile)
                      for ns, nb in rows])

        return pd.DataFrame({
            "n_src": rows[:, 0],
            "n_bkg": rows[:, 1],
            f"predicted_time_{_qkey(quantile)}": t,
        })


    def model_info(self, quantile=0.99):
        """Return name, formula, R2, and params for the best model at a quantile."""
        k = _qkey(quantile)
        if k not in self._models:
            raise ValueError(
                f"Quantile {quantile} not directly fitted. "
                f"Available: {self.quantiles}")
        m = self._models[k]
        return dict(name=m["name"], formula=m["formula"],
                    r2=m["r2"], params=m["params"])

    # Diagnostic plots
    def plot_bin_counts(self, output_path=None, title="Maps per bin"):
        """
        Heatmap of sample counts per (n_src, n_bkg) bin.
        Uses the first fitted quantile's binned data — all quantiles
        share the same bin structure so any key gives the same counts.
        """
        k                = self._qkeys[0]
        bin_src, bin_bkg = self._binned[k]["X"]
        counts           = self._binned[k]["counts"]

        src_centers = 0.5 * (self._src_edges[:-1] + self._src_edges[1:])
        bkg_centers = 0.5 * (self._bkg_edges[:-1] + self._bkg_edges[1:])

        grid = np.full((len(bkg_centers), len(src_centers)), np.nan)
        for s, b, n in zip(bin_src, bin_bkg, counts):
            si = int(np.argmin(np.abs(src_centers - s)))
            bi = int(np.argmin(np.abs(bkg_centers - b)))
            grid[bi, si] = n

        masked = np.ma.masked_where(np.isnan(grid), grid)

        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.pcolormesh(self._src_edges, self._bkg_edges, masked,
                        cmap="viridis", shading="auto")
        plt.colorbar(im, ax=ax, label="Count per bin")

        for s, b, n in zip(bin_src, bin_bkg, counts):
            ax.text(s, b, str(int(n)), ha="center", va="center",
                    fontsize=6, color="white")

        ax.set_xlabel("n_src")
        ax.set_ylabel("n_bkg")
        ax.set_title(title)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_surface(self, quantile=0.99, output_path=None):
        """3D surface of the best model for one quantile."""
        k = _qkey(quantile)
        if k not in self._models:
            raise ValueError(f"Quantile {quantile} not directly fitted.")
        m = self._models[k]

        n_grid = 60
        sg = np.linspace(max(self._n_src_raw.min(), 1), self._n_src_raw.max(), n_grid)
        bg = np.linspace(max(self._n_bkg_raw.min(), 1), self._n_bkg_raw.max(), n_grid)
        SG, BG = np.meshgrid(sg, bg)
        Z = m["func"]((SG.ravel(), BG.ravel()), *m["params"]).reshape(SG.shape)

        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot_surface(SG, BG, Z, alpha=0.45, cmap="coolwarm",
                        edgecolor="none", rstride=3, cstride=3)

        X_bin = self._binned[k]["X"]
        y_bin = self._binned[k]["y"]
        ax.scatter(X_bin[0], X_bin[1], y_bin, s=25, color="black",
                   edgecolors="white", linewidths=0.4, zorder=10)

        ax.set_xlabel("n_src"); ax.set_ylabel("n_bkg"); ax.set_zlabel("time (s)")
        ax.set_title(f"q={quantile}  {m['name']}  R2={m['r2']:.4f}")
        ax.view_init(elev=25, azim=-50)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_pred_vs_target(self, output_path=None):
        """
        Predicted vs binned target scatter for the best model per quantile.
        Dot size is proportional to bin sample count.
        """
        cmap   = plt.cm.plasma
        colors = {k: cmap(i / max(len(self._qkeys) - 1, 1))
                for i, k in enumerate(self._qkeys)}

        ncols = min(len(self.quantiles), 4)
        nrows = int(np.ceil(len(self.quantiles) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(5 * ncols, 4.5 * nrows),
                                squeeze=False)
        axes_flat = axes.flatten()

        for i, (q, k) in enumerate(zip(self.quantiles, self._qkeys)):
            if k not in self._models:
                continue
            ax     = axes_flat[i]
            m      = self._models[k]
            X_bin  = self._binned[k]["X"]
            y_bin  = self._binned[k]["y"]
            counts = self._binned[k]["counts"]
            y_pred = m["func"](X_bin, *m["params"])

            ax.scatter(y_bin, y_pred, s=counts * 2, alpha=0.7,
                    c=colors[k], edgecolors="k", linewidths=0.3)
            lims = [0, max(y_bin.max(), y_pred.max()) * 1.1]
            ax.plot(lims, lims, "k--", lw=1)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel("Binned Target (s)", fontsize=9)
            ax.set_ylabel("Predicted (s)", fontsize=9)
            ax.set_title(f"p={q}: {m['name']}\nR²={m['r2']:.4f}", fontsize=9)
            ax.tick_params(labelsize=8)

        for j in range(len(self.quantiles), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Pred vs Binned Target — Mapping Time Quantiles\n"
                    "(dot size ∝ bin count)", fontsize=13, y=1.01)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()




if __name__ == "__main__":
    #example usage
    # from mapping_time_query import MapTimeQueryEngine
    engine = MapTimeQueryEngine(
        csv_path="emsoft.csv",
        quantiles=[0.90, 0.99, 0.999],
        n_bins_per_axis=20,
        min_bin_count=10,
    )

    # Single query
    n_src, n_bkg = 400, 900
    t = engine.query(n_src=n_src, n_bkg=n_bkg, quantile=0.99)
    print(f"\nPredicted mapping time (q=0.99 | n_src={n_src}, n_bkg={n_bkg}): {t:.3f} s")

    # linearly interpolated query (if 0.995 is not fitted)
    t2 = engine.query(n_src=n_src, n_bkg=n_bkg, quantile=0.995)
    print(f"Predicted mapping time (q=0.995, interpolated): {t2:.3f} s")

    # batch query
    print("\nBatch query (q=0.99):")
    queries = pd.DataFrame([
        {"n_src": 200, "n_bkg":  500},
        {"n_src": 400, "n_bkg":  900},
        {"n_src": 600, "n_bkg": 1200},
    ])
    predicts=engine.batch_query(queries, quantile=0.99)
    print(predicts.to_string(index=False))

    # diagnose
    info = engine.model_info(quantile=0.99)
    print(f"\nBest model (q=0.99): {info['name']}  R2={info['r2']:.4f}")
    print(f"  Formula : {info['formula']}")
    print(f"  Params  : {[f'{p:.6f}' for p in info['params']]}")

    engine.plot_surface(quantile=0.99, output_path="map_surface_q99.png")
    engine.plot_pred_vs_target(output_path="map_predvs.png")
    engine.plot_bin_counts(output_path="map_bin_counts.png")
    print("\nPlots saved.")
