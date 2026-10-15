"""Paper B: evaluating BDP Model Gate as a pre-release review tool for a late-delivery
model, against three comparators, on a battery of independently labelled cases with
known ground truth (clean reference, four defect types, two legitimate change
scenarios), per model family, per dataset.

The operational use case throughout: reviewing a late-delivery model before it is
released to score future orders. The reference pipeline (the "clean" case) is the
chronological, order/customer-safe pipeline already established and frozen in Paper A
(Section O2): time-aware split, leakage-free predictors, held-out final period, a
training configuration (hyperparameters) frozen from that paper's own tuning, never
re-tuned against this paper's own evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .metrics import all_metrics, safe_auc
from .models import build_preprocessor, make_estimator

MODEL_FAMILIES = ["logreg", "rf", "gbm"]

# Frozen hyperparameters, taken verbatim from Paper A's O2 chronological-primary tuning
# (results/o2_design_comparison/*_results.csv, design=chronological_primary). Never
# re-tuned against this study's own cases.
FROZEN_PARAMS = {
    "logreg": {"C": 0.1},
    "rf": {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 20},
    "gbm": {"n_estimators": 150, "num_leaves": 15, "learning_rate": 0.1},
}

# Frozen decision thresholds for the non-gate comparators, set from the clean
# reference cases' own metric distributions and never adjusted after seeing any
# defect or scenario case's result.
AUROC_ONLY_THRESHOLD = 0.65
METRICS_ONLY_AUROC_THRESHOLD = 0.65
METRICS_ONLY_BRIER_THRESHOLD = 0.22
METRICS_ONLY_CAL_SLOPE_BOUNDS = (0.7, 1.4)

# Frozen extended-check and automated-checklist overlap thresholds (same values
# used for both, since both compute an overlap fraction and are held to the same
# standard rather than one being quietly more lenient than the other).
GROUP_OVERLAP_NEEDS_REVIEW = 0.01
GROUP_OVERLAP_BLOCKED = 0.05

# Known, defect-independent baseline issues: (dataset, model) pairs whose
# chronological reference pipeline the companion validation-design study already
# documented as broken for a reason unrelated to any defect this paper injects
# (Olist's random forest collapses under a 2018 service-promise contraction).
# A verdict on one of these is a defensible block for a real, pre-existing reason,
# not evidence either for or against detection of the defect under test, and is
# reported separately rather than counted as a false alarm or as detection.
KNOWN_BASELINE_ISSUE = {("olist", "rf")}


def fit_predict(family, X_train, y_train, X_test, numeric_cols, categorical_cols, seed=0):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline([("prep", clone(preprocessor)),
                      ("clf", make_estimator(family, FROZEN_PARAMS[family], seed=seed))])
    pipe.fit(X_train, y_train)
    p_test = pipe.predict_proba(X_test)[:, 1]
    return pipe, p_test


def auroc_only_rule(auc):
    return "BLOCKED" if (np.isnan(auc) or auc < AUROC_ONLY_THRESHOLD) else "PASS"


def metrics_only_rule(auc, brier, cal_slope):
    """A metrics-only reviewer (previously, and inaccurately, called a "checklist"):
    computed discrimination, calibration and loss metrics against frozen thresholds,
    with no schema check, no overlap check and no provenance check of any kind -- it
    can only react to what shows up in the three metrics themselves, which is the
    point being tested (leakage typically improves, not worsens, these metrics).
    """
    if np.isnan(auc) or auc < METRICS_ONLY_AUROC_THRESHOLD or (not np.isnan(brier) and brier > METRICS_ONLY_BRIER_THRESHOLD):
        return "BLOCKED"
    lo, hi = METRICS_ONLY_CAL_SLOPE_BOUNDS
    if np.isnan(cal_slope) or cal_slope < lo or cal_slope > hi:
        return "NEEDS_REVIEW"
    return "PASS"


def automated_checklist_rule(auc, brier, cal_slope, X_train, X_test):
    """A more capable, but still simple and non-bootstrapped, automated checklist: the
    metrics-only rule above, plus two checks a conscientious team can write without a
    dedicated governance library -- a schema check (do training and review-time
    columns match, in the same order) and an exact-row-overlap check (what fraction of
    scored rows' full feature vectors also appear, unchanged, in training), computed
    directly via a row hash rather than any bootstrap. This is the comparator a team
    that has gone slightly beyond "read the metrics" but has not adopted a dedicated
    governance tool would plausibly run.
    """
    if list(X_train.columns) != list(X_test.columns):
        return "BLOCKED"
    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False).values)
    test_hashes = pd.util.hash_pandas_object(X_test, index=False).values
    overlap_frac = float(np.mean(np.isin(test_hashes, list(train_hashes)))) if len(test_hashes) else 0.0
    metrics_verdict = metrics_only_rule(auc, brier, cal_slope)
    if metrics_verdict == "BLOCKED" or overlap_frac >= GROUP_OVERLAP_BLOCKED:
        return "BLOCKED"
    if metrics_verdict == "NEEDS_REVIEW" or overlap_frac >= GROUP_OVERLAP_NEEDS_REVIEW:
        return "NEEDS_REVIEW"
    return "PASS"


def temporal_integrity_rule(time_train, cutoff):
    """A minimal, identifier-free comparator: does any training row's own decision
    time fall on or after the release cutoff (the start of the held-out scoring
    window)? This needs no declared group identifier at all, only the decision-time
    column every case already carries, and exists to establish what a basic
    chronological-ordering check alone already buys a reviewer -- before crediting
    the extended group-identifier check (`group_overlap_check`) with detection a
    much simpler, identifier-free check would have caught regardless. A hard
    temporal fact rather than a graduated risk, so it has no NEEDS_REVIEW tier:
    a training row timestamped at or after the cutoff is never legitimate.
    """
    time_train = np.asarray(time_train)
    violation = time_train >= np.datetime64(cutoff)
    n_violations = int(violation.sum())
    status = "BLOCKED" if n_violations > 0 else "PASS"
    return {"check_name": "temporal_integrity", "status": status,
            "n_violations": n_violations, "n_train": len(time_train)}


def group_overlap_check(group_train, group_test, time_train=None, test_period_start=None):
    """The extended check this paper adds to the stock gate: explicit train/test
    overlap measured on a declared *group* identifier, not on feature-level row
    duplicates. Motivated directly by the companion study's finding that the stock
    gate's SplitOverlapCheck operates at the feature/row level and structurally
    cannot see group-level (non-independence) leakage when sibling rows or sibling
    records do not share an identical feature vector.

    This same function, with the same declared identifier and the same policy, is
    applied to every case in the battery -- clean, every defect, and both
    scenarios -- never selected conditional on which case is being scored.

    When `time_train` and `test_period_start` are supplied, a training-side group
    member only counts as overlap if its own decision time falls on or after
    `test_period_start` (contemporaneous-or-future leakage); a group's genuinely
    past history in training is not flagged, consistent with the companion study's
    finding (O3a, condition D) that legitimate customer history is not a leakage
    channel. Without them, any shared group id counts as overlap, which is the
    correct policy when the group is an order id: an order has a single decision
    time, so if any of its rows are in training at all while it is being scored,
    that is illegitimate regardless of when the overlap occurs.
    """
    test_groups = set(pd.unique(group_test))
    if time_train is not None and test_period_start is not None:
        illegitimate_mask = time_train >= np.datetime64(test_period_start)
        train_groups = set(pd.unique(np.asarray(group_train)[illegitimate_mask]))
    else:
        train_groups = set(pd.unique(group_train))
    overlap = train_groups & test_groups
    frac = len(overlap) / max(len(test_groups), 1)
    if frac >= GROUP_OVERLAP_BLOCKED:
        status = "BLOCKED"
    elif frac >= GROUP_OVERLAP_NEEDS_REVIEW:
        status = "NEEDS_REVIEW"
    else:
        status = "PASS"
    return {"check_name": "group_overlap_extended", "status": status,
            "overlap_fraction": frac, "n_overlapping_groups": len(overlap),
            "n_test_groups": len(test_groups)}


def combine_verdicts(*statuses):
    """Most severe status wins: BLOCKED > NEEDS_REVIEW > PASS."""
    order = {"PASS": 0, "NEEDS_REVIEW": 1, "BLOCKED": 2, "ERROR": 2}
    return max(statuses, key=lambda s: order.get(s, 1))
