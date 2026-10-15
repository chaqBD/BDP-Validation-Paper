"""The five validation designs compared in O2 (Section 5.1), plus rolling-origin
splits reused by O4 (Section 7.4).

`meta` is the per-row DataFrame with columns order_id, customer_id, decision_time
produced by dataco_prep.build_population / olist_prep.build_population. Index
positions in meta line up with positions in X and y.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def split_row_random(meta, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(meta)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    return idx[n_test:], idx[:n_test]


def split_group_random(meta, group_col, test_size=0.2, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(meta, groups=meta[group_col]))
    return train_idx, test_idx


def _group_contiguous_order(decision_time, group_id):
    """An ordering by decision_time in which rows sharing a group id are always
    contiguous, including when two *different* groups are tied on decision time.

    A plain `argsort(decision_time)` does not guarantee this: a stable sort merely
    preserves each tied group's original storage order, so two different groups
    tied on the same timestamp can end up interleaved (e.g. A, B, A, B) if their
    rows were not already stored grouped together, and a cut landing inside that
    interleaving would split group A's rows across train and test even though
    adjacent positions never literally repeat a group id back-to-back -- the
    failure mode an earlier, adjacency-only boundary check here did not catch.
    Sorting by (decision_time, group_id) as a composite key removes the
    ambiguity: within a decision-time tie, rows are grouped by id first, so every
    group's rows are contiguous by construction, not by chance.
    """
    group_codes = np.unique(group_id, return_inverse=True)[1]
    return np.lexsort((group_codes, decision_time))


def split_chronological(meta, test_size=0.2):
    group_id = meta["order_id"].values if "order_id" in meta.columns else np.arange(len(meta))
    order = _group_contiguous_order(meta["decision_time"].values, group_id)
    group_sorted = group_id[order]
    n = len(meta)
    n_test_raw = int(n * test_size)
    test_start = _snap_forward_to_group_boundary(n - n_test_raw, group_sorted)
    test_pos = order[test_start:]
    train_pos = order[:test_start]
    cutoff = meta["decision_time"].values[test_pos].min()
    assert len(np.intersect1d(group_id[train_pos], group_id[test_pos])) == 0, \
        "split_chronological produced overlapping group ids between train and test"
    return train_pos, test_pos, cutoff


def split_chronological_new_customers(meta, test_size=0.2):
    train_idx, test_idx, cutoff = split_chronological(meta, test_size=test_size)
    train_customers = set(meta["customer_id"].values[train_idx])
    test_customer_ids = meta["customer_id"].values[test_idx]
    is_new = np.array([c not in train_customers for c in test_customer_ids])
    return train_idx, test_idx[is_new], test_idx[~is_new], cutoff


def match_training_size(train_idx_dict, meta, group_col="order_id", seed=0):
    """Subsample each design's training set down to the smallest common number of
    unique orders across designs, so comparisons are not confounded by training
    size (Section 7.1 'Equal footing').
    """
    rng = np.random.default_rng(seed)
    n_orders = {k: meta[group_col].values[idx].astype(str)
                for k, idx in train_idx_dict.items()}
    min_orders = min(len(np.unique(v)) for v in n_orders.values())

    matched = {}
    for k, idx in train_idx_dict.items():
        groups = meta[group_col].values[idx].astype(str)
        unique_groups = np.unique(groups)
        if len(unique_groups) <= min_orders:
            matched[k] = idx
            continue
        keep_groups = set(rng.choice(unique_groups, size=min_orders, replace=False))
        keep_mask = np.isin(groups, list(keep_groups))
        matched[k] = idx[keep_mask]
    return matched, min_orders


def _snap_forward_to_group_boundary(pos, group_sorted):
    """Advance `pos` until it does not fall strictly inside a run of identical
    group ids in `group_sorted` (e.g. two different orders tied on decision time,
    with their rows interleaved in storage order, must not be split by a cut that
    lands between two of the same order's rows). A no-op when every position is
    already its own group, as for a row = order dataset."""
    n = len(group_sorted)
    while 0 < pos < n and group_sorted[pos] == group_sorted[pos - 1]:
        pos += 1
    return min(pos, n)


def rolling_origin_splits(meta, n_origins=6, min_train_frac=0.5, test_frac_per_origin=0.08):
    """Expanding-window rolling-origin splits (Section 7.4).

    Boundaries are snapped so that no group (order id) is ever split across the
    train/test cut, even when two different groups are tied on decision time and
    happen to be interleaved in storage order within that tie.

    Returns a list of dicts with train_idx, test_idx, origin_time (start of the
    test window) and horizon_end (end of the test window).
    """
    group_id = meta["order_id"].values if "order_id" in meta.columns else np.arange(len(meta))
    order = _group_contiguous_order(meta["decision_time"].values, group_id)
    times_sorted = meta["decision_time"].values[order]
    group_sorted = group_id[order]
    n = len(order)

    min_train_n = int(n * min_train_frac)
    test_n = max(int(n * test_frac_per_origin), 1)
    remaining = n - min_train_n
    step = max(remaining // n_origins, 1)

    splits = []
    start = min_train_n
    for i in range(n_origins):
        train_end = _snap_forward_to_group_boundary(start + i * step, group_sorted)
        test_start = train_end
        test_end = _snap_forward_to_group_boundary(min(test_start + test_n, n), group_sorted)
        if test_start >= n or test_end - test_start < 20:
            break
        train_pos = order[:train_end]
        test_pos = order[test_start:test_end]
        assert len(np.intersect1d(group_id[train_pos], group_id[test_pos])) == 0, \
            f"rolling_origin_splits origin {i} produced overlapping group ids between train and test"
        splits.append({
            "origin_index": i,
            "train_idx": train_pos,
            "test_idx": test_pos,
            "origin_time": times_sorted[test_start],
            "horizon_end": times_sorted[test_end - 1],
            "train_end_time": times_sorted[train_end - 1],
        })
    return splits


def weekly_block_ids(decision_time):
    """ISO week identifiers used as cluster ids for block bootstrap on
    chronological comparisons (Section 7.5).
    """
    import pandas as pd
    ts = pd.to_datetime(decision_time)
    iso = ts.isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).values
