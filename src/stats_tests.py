"""RQ-specific tests: ranking stability (H6) and dose-response slope (H3)."""
import numpy as np
from scipy.stats import kendalltau, linregress


def kendall_tau_with_bootstrap(ranks_a, ranks_b, n_boot=1000, seed=0, ci=0.95):
    tau, pvalue = kendalltau(ranks_a, ranks_b)
    rng = np.random.default_rng(seed)
    n = len(ranks_a)
    ranks_a = np.asarray(ranks_a)
    ranks_b = np.asarray(ranks_b)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        t, _ = kendalltau(ranks_a[idx], ranks_b[idx])
        boots[b] = t
    boots = boots[~np.isnan(boots)]
    alpha = 1 - ci
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2]) if len(boots) else (np.nan, np.nan)
    return {"tau": tau, "pvalue": pvalue, "ci_low": lo, "ci_high": hi,
            "rank_change_frequency": float(np.mean(ranks_a != ranks_b))}


def slope_test(x, y):
    """Simple OLS slope with its two-sided p-value (used for H3's dose-response)."""
    res = linregress(x, y)
    return {"slope": res.slope, "intercept": res.intercept, "pvalue": res.pvalue,
            "r_value": res.rvalue, "stderr": res.stderr}


def population_stability_index(expected, actual, n_bins=10):
    """PSI between a reference and a later distribution of a numeric feature."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    quantiles = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    quantiles = np.unique(quantiles)
    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def standardized_mean_shift(expected, actual):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    pooled_sd = np.sqrt((np.nanvar(expected) + np.nanvar(actual)) / 2)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return float((np.mean(actual) - np.mean(expected)) / pooled_sd)
