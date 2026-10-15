"""Constructs the independently labelled review cases for each dataset: one clean
reference, four defect types with known ground truth, and two legitimate
distribution-change scenarios (not defects). All cases share the same reference
pipeline's chronological train/test boundary and training size wherever the defect
construction does not itself require changing that boundary.
"""
import numpy as np
import pandas as pd

from . import dataco_prep as dc
from . import olist_prep as ol
from .o2_pipeline import build_designs
from .splitters import rolling_origin_splits

RNG_SEED = 20261015


def _reference(dataset):
    if dataset == "dataco":
        df = dc.load_raw()
        X, y, meta, num, cat = dc.build_population(df)
        group_col = "order_id"
    else:
        orders, customers, items, payments, products = ol.load_raw()
        X, y, meta, num, cat = ol.build_population(orders, customers, items, payments, products)
        group_col = "order_id"  # Olist row = order; "related-order overlap" is customer-level
    designs, _ = build_designs(meta, test_size=0.2, seed=RNG_SEED)
    train_idx, test_idx = designs["chronological_primary"]
    return {"X": X, "y": y, "meta": meta, "num": num, "cat": cat,
            "train_idx": train_idx, "test_idx": test_idx, "group_col": group_col}


def case_clean(ref):
    return {"train_idx": ref["train_idx"], "test_idx": ref["test_idx"],
            "X": ref["X"], "extra_feature": None}


def case_leaky_feature(ref, dataset):
    """Defect: a post-outcome predictor (actual fulfilment time) is mistakenly left in
    the feature set. Ground truth: target leakage present.
    """
    X = ref["X"].copy()
    X["leaky_actual_fulfillment_days"] = ref["meta"]["actual_fulfillment_days"].values
    num2 = ref["num"] + ["leaky_actual_fulfillment_days"]
    return {"train_idx": ref["train_idx"], "test_idx": ref["test_idx"], "X": X,
            "num_override": num2}


def case_exact_duplicate(ref, frac=0.2, seed=RNG_SEED):
    """Defect: a fraction of the held-out test rows are also present, unchanged, in
    the training set (e.g. an accidental re-merge or a failed dedup step upstream).
    """
    rng = np.random.default_rng(seed)
    test_idx = ref["test_idx"]
    n_dup = max(int(len(test_idx) * frac), 1)
    dup_rows = rng.choice(test_idx, size=n_dup, replace=False)
    train_idx = np.concatenate([ref["train_idx"], dup_rows])
    return {"train_idx": train_idx, "test_idx": test_idx, "X": ref["X"],
            "n_duplicated": n_dup}


def case_group_overlap_dataco(ref, seed=RNG_SEED):
    """Defect, DataCo: sibling line items of test orders are present in training,
    while the *scored* test set is reduced to exactly one kept representative row
    per order -- the sibling rows that move to training are removed from the
    scored set, so no row index is ever both trained on and scored (distinct-row
    group-dependence leakage, not exact row duplication, which is a separate,
    already-covered defect, `case_exact_duplicate`).
    """
    order_id = ref["meta"]["order_id"].values
    test_idx = ref["test_idx"]
    test_orders_df = pd.DataFrame({"pos": test_idx, "order_id": order_id[test_idx]})
    grouped = test_orders_df.groupby("order_id")["pos"].apply(lambda s: s.values)
    kept_pos = np.array([v[0] for v in grouped.values])
    sibling_pos = np.concatenate([v[1:] for v in grouped.values if len(v) > 1]) \
        if any(len(v) > 1 for v in grouped.values) else np.array([], dtype=int)
    train_idx = np.concatenate([ref["train_idx"], sibling_pos])
    assert len(np.intersect1d(train_idx, kept_pos)) == 0, \
        "group-overlap case must not re-score a row used in training"
    return {"train_idx": train_idx, "test_idx": kept_pos, "X": ref["X"],
            "n_leaked_siblings": len(sibling_pos), "n_test_orders": len(kept_pos)}


def case_group_overlap_olist(ref, seed=RNG_SEED):
    """Defect, Olist: a customer with more than one order inside the held-out test
    window has those other, same-window orders present in training too (illegitimate
    -- distinct from a customer's genuinely past history, which O3a's condition D
    already shows is not a leakage channel). The scored test set keeps exactly one
    (the earliest) order per such customer; the customer's other same-window
    orders move to training and are removed from the scored set, so no row index
    is ever both trained on and scored.
    """
    customer_id = ref["meta"]["customer_id"].values
    decision_time = ref["meta"]["decision_time"].values
    test_idx = ref["test_idx"]
    df = pd.DataFrame({"pos": test_idx, "customer_id": customer_id[test_idx],
                        "decision_time": decision_time[test_idx]}).sort_values("decision_time")
    grouped = df.groupby("customer_id")["pos"].apply(lambda s: s.values)
    kept_pos = np.array([v[0] for v in grouped.values])
    extra_pos = np.concatenate([v[1:] for v in grouped.values if len(v) > 1]) \
        if any(len(v) > 1 for v in grouped.values) else np.array([], dtype=int)
    train_idx = np.concatenate([ref["train_idx"], extra_pos])
    assert len(np.intersect1d(train_idx, kept_pos)) == 0, \
        "group-overlap case must not re-score a row used in training"
    return {"train_idx": train_idx, "test_idx": kept_pos, "X": ref["X"],
            "n_leaked_same_window_orders": len(extra_pos), "n_test_orders": len(kept_pos)}


def case_invalid_schema(ref):
    """Defect: the data handed to the reviewing process has a schema mismatch
    relative to what the model was fitted on (a dropped column, simulating a broken
    feature-pipeline handoff). Model predictions are computed from the correct,
    full schema beforehand; only the review-time X is perturbed.
    """
    drop_col = ref["num"][0]
    return {"train_idx": ref["train_idx"], "test_idx": ref["test_idx"], "X": ref["X"],
            "drop_col_for_review": drop_col}


def case_scenario(dataset, origin_index, n_origins=10, min_train_frac=0.4,
                   test_frac_per_origin=0.06):
    """A legitimate distribution-change scenario (not a defect): a rolling-origin
    window identified in O4 as high-drift or high-prevalence-change. Uses the exact
    same origin construction as O4 so the window matches that paper's own findings.
    """
    if dataset == "dataco":
        df = dc.load_raw()
        X, y, meta, num, cat = dc.build_population(df)
    else:
        orders, customers, items, payments, products = ol.load_raw()
        X, y, meta, num, cat = ol.build_population(orders, customers, items, payments, products)
    splits = rolling_origin_splits(meta, n_origins=n_origins, min_train_frac=min_train_frac,
                                    test_frac_per_origin=test_frac_per_origin)
    s = splits[origin_index]
    return {"train_idx": s["train_idx"], "test_idx": s["test_idx"], "X": X, "y": y,
            "meta": meta, "num": num, "cat": cat, "origin_time": str(s["origin_time"])}
