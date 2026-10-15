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

# Frozen decision thresholds for the two non-gate comparators, set from the clean
# reference cases' own metric distributions and never adjusted after seeing any
# defect or scenario case's result.
AUROC_ONLY_THRESHOLD = 0.65
CHECKLIST_AUROC_THRESHOLD = 0.65
CHECKLIST_BRIER_THRESHOLD = 0.22
CHECKLIST_CAL_SLOPE_BOUNDS = (0.7, 1.4)

# Frozen extended-check thresholds (group-overlap fraction between train and test).
GROUP_OVERLAP_NEEDS_REVIEW = 0.01
GROUP_OVERLAP_BLOCKED = 0.05


def fit_predict(family, X_train, y_train, X_test, numeric_cols, categorical_cols, seed=0):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline([("prep", clone(preprocessor)),
                      ("clf", make_estimator(family, FROZEN_PARAMS[family], seed=seed))])
    pipe.fit(X_train, y_train)
    p_test = pipe.predict_proba(X_test)[:, 1]
    return pipe, p_test


def auroc_only_rule(auc):
    return "BLOCKED" if (np.isnan(auc) or auc < AUROC_ONLY_THRESHOLD) else "PASS"


def checklist_rule(auc, brier, cal_slope):
    """A standard metrics-and-checklist reviewer: computed discrimination, calibration
    and loss metrics against frozen thresholds. Deliberately has no leakage-specific
    computation -- it can only react to what shows up in the metrics themselves, which
    is the point being tested (leakage typically improves, not worsens, these metrics).
    """
    if np.isnan(auc) or auc < CHECKLIST_AUROC_THRESHOLD or (not np.isnan(brier) and brier > CHECKLIST_BRIER_THRESHOLD):
        return "BLOCKED"
    lo, hi = CHECKLIST_CAL_SLOPE_BOUNDS
    if np.isnan(cal_slope) or cal_slope < lo or cal_slope > hi:
        return "NEEDS_REVIEW"
    return "PASS"


def group_overlap_check(group_train, group_test):
    """The extended check this paper adds to the stock gate: explicit train/test
    overlap measured on a declared *group* identifier (order id / customer id), not on
    feature-level row duplicates. Motivated directly by Paper A's finding (Section 6.6)
    that the stock gate's SplitOverlapCheck operates at the feature/row level and
    structurally cannot see group-level (non-independence) leakage when sibling rows
    or sibling records do not share an identical feature vector.
    """
    train_groups = set(pd.unique(group_train))
    test_groups = set(pd.unique(group_test))
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
