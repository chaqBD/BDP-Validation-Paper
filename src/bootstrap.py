"""Uncertainty estimation that respects dependence (Section 7.5).

Cluster bootstrap resamples whole clusters (orders / customers / weekly blocks),
not rows, so that dependent observations move together across resamples
(Field & Welsh, 2007; Kunsch, 1989).
"""
import numpy as np


def cluster_bootstrap_metric(y_true, y_prob, cluster_ids, metric_fn, n_boot=500, seed=0,
                              ci=0.95):
    """Bootstrap a scalar metric by resampling clusters with replacement.

    Returns (point_estimate, ci_low, ci_high, boot_samples).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    cluster_ids = np.asarray(cluster_ids)
    unique_clusters = np.unique(cluster_ids)

    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    point = metric_fn(y_true, y_prob)

    boots = np.empty(n_boot)
    n_clusters = len(unique_clusters)
    for b in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])
        boots[b] = metric_fn(y_true[idx], y_prob[idx])

    boots = boots[~np.isnan(boots)]
    alpha = 1 - ci
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2]) if len(boots) else (np.nan, np.nan)
    return point, lo, hi, boots


def paired_cluster_bootstrap_difference(y_true_a, y_prob_a, cluster_ids_a, metric_fn,
                                         y_true_b, y_prob_b, cluster_ids_b,
                                         n_boot=500, seed=0, ci=0.95):
    """Bootstrap the difference metric(A) - metric(B) for two (possibly different-sized)
    evaluations sharing the same underlying cluster universe is not assumed; each side is
    resampled independently at the cluster level (used when A and B are evaluated on
    different validation-design samples of a shared order/customer population), following
    the paired-bootstrap approach of Section 7.5.
    """
    rng = np.random.default_rng(seed)
    pa, _, _, boots_a = cluster_bootstrap_metric(y_true_a, y_prob_a, cluster_ids_a, metric_fn,
                                                  n_boot=n_boot, seed=seed, ci=ci)
    pb, _, _, boots_b = cluster_bootstrap_metric(y_true_b, y_prob_b, cluster_ids_b, metric_fn,
                                                  n_boot=n_boot, seed=seed + 1, ci=ci)
    diffs = boots_a - boots_b
    diffs = diffs[~np.isnan(diffs)]
    alpha = 1 - ci
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2]) if len(diffs) else (np.nan, np.nan)
    return pa - pb, lo, hi, diffs


def block_bootstrap_metric(y_true, y_prob, block_ids, metric_fn, n_boot=500, seed=0, ci=0.95):
    """Moving/weekly block bootstrap for chronological comparisons (Kunsch, 1989)."""
    return cluster_bootstrap_metric(y_true, y_prob, block_ids, metric_fn, n_boot=n_boot,
                                     seed=seed, ci=ci)


def equivalence_test(diff, ci_low, ci_high, delta):
    """Classify a paired difference against an equivalence margin (Lakens, 2017).

    Returns one of: 'superior', 'inferior', 'equivalent', 'inconclusive'.
    """
    if ci_low > delta:
        return "superior"
    if ci_high < -delta:
        return "inferior"
    if ci_low > -delta and ci_high < delta:
        return "equivalent"
    return "inconclusive"


def holm_adjust(pvalues, alpha=0.05):
    """Holm's step-down adjustment. Returns adjusted p-values in the original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted
