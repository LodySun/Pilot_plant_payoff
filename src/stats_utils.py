"""Group statistics: Wilcoxon signed-rank with effect sizes, and FDR.

Kept separate so Stage 3 and Stage 4 report numbers the same way.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_vs(values: np.ndarray, null_value: float = 0.0) -> dict:
    """One-sample Wilcoxon signed-rank against a reference value.

    Returns the statistic, a two-sided p, the matched-pairs rank-biserial
    correlation as the effect size, and the descriptives a reader needs to
    judge the result. Degenerate cases (too few subjects, all differences
    exactly zero) return NaN p rather than raising.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    out = {
        "n": n,
        "mean": float(np.mean(v)) if n else np.nan,
        "median": float(np.median(v)) if n else np.nan,
        "sd": float(np.std(v, ddof=1)) if n > 1 else np.nan,
        "null_value": null_value,
        "statistic": np.nan,
        "p_value": np.nan,
        "rank_biserial": np.nan,
        "cohens_d": np.nan,
        "n_above": int(np.sum(v > null_value)) if n else 0,
        "n_below": int(np.sum(v < null_value)) if n else 0,
    }
    if n < 3:
        return out

    d = v - null_value
    if np.all(d == 0):
        out["statistic"] = 0.0
        out["p_value"] = 1.0
        out["rank_biserial"] = 0.0
        out["cohens_d"] = 0.0
        return out

    nz = d[d != 0]
    try:
        res = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        out["statistic"] = float(res.statistic)
        out["p_value"] = float(res.pvalue)
    except ValueError:
        return out

    # Rank-biserial: (W+ - W-) / (W+ + W-), signed so positive means
    # values sit above the reference.
    ranks = stats.rankdata(np.abs(nz))
    w_plus = float(np.sum(ranks[nz > 0]))
    w_minus = float(np.sum(ranks[nz < 0]))
    total = w_plus + w_minus
    out["rank_biserial"] = (w_plus - w_minus) / total if total else 0.0

    sd = np.std(d, ddof=1)
    out["cohens_d"] = float(np.mean(d) / sd) if sd > 0 else np.nan
    return out


def fdr(p_values, alpha: float = 0.05, method: str = "fdr_bh"):
    """Benjamini-Hochberg FDR across a set of tests, NaNs passed through."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(p_values, dtype=float)
    out_p = np.full(p.shape, np.nan)
    out_rej = np.zeros(p.shape, dtype=bool)
    good = np.isfinite(p)
    if good.sum() == 0:
        return out_rej, out_p
    rej, corrected, _, _ = multipletests(p[good], alpha=alpha, method=method)
    out_p[good] = corrected
    out_rej[good] = rej
    return out_rej, out_p


def star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# Moved from run_zadbood_replication.py (unchanged).
def directional_wilcoxon(values, null: float, alternative: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    diff = values - float(null)
    if values.size < 3:
        p = np.nan
        statistic = np.nan
    elif np.all(diff == 0):
        p, statistic = 1.0, 0.0
    else:
        result = stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox")
        p, statistic = float(result.pvalue), float(result.statistic)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "n_above": int(np.sum(values > null)),
        "n_below": int(np.sum(values < null)),
        "statistic": statistic,
        "p_value": p,
        "alternative": alternative,
        "null_value": float(null),
    }
