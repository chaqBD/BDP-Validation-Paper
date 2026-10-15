"""Tests directly, rather than asserting, whether the group-identifier extension
is actually resilient to a corrupted or backdated training-timestamp column -- an
advantage an earlier draft of this paper claimed for the extension in general,
without noting that the claim only has a basis for DataCo's policy.

DataCo's declared policy (Section 4.5) uses no time input at all: any shared
order id between train and test counts as overlap regardless of when either row
is dated. Olist's declared policy is explicitly time-aware (a training order only
counts as overlap if its own decision time falls on or after the held-out
window's start), which means it is just as dependent on a trustworthy timestamp
column as the temporal-integrity check it is being compared against.

This script corrupts every training row's decision time to an arbitrarily early,
safely-in-the-past value (simulating a backdated or corrupted timestamp column)
and re-runs both the extension and the temporal-integrity check on the
group-overlap defect for each dataset, under its own declared policy.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.release_cases import _reference, case_group_overlap_dataco, case_group_overlap_olist
from src.release_review import group_overlap_check, temporal_integrity_rule

OUT_DIR = RESULTS_DIR / "release_review"


def run(dataset, case_builder, identifier_col, time_aware):
    ref = _reference(dataset)
    case = case_builder(ref)
    meta = ref["meta"]
    train_idx, test_idx = case["train_idx"], case["test_idx"]
    group_train = meta[identifier_col].values[train_idx]
    group_test = meta[identifier_col].values[test_idx]
    cutoff = meta["decision_time"].values[test_idx].min()
    corrupted_time_train = np.full(len(train_idx), meta["decision_time"].values.min())

    if time_aware:
        real_time_train = meta["decision_time"].values[train_idx]
        extension_uncorrupted = group_overlap_check(group_train, group_test, time_train=real_time_train,
                                                      test_period_start=cutoff)
        extension_corrupted = group_overlap_check(group_train, group_test, time_train=corrupted_time_train,
                                                    test_period_start=cutoff)
    else:
        extension_uncorrupted = group_overlap_check(group_train, group_test)
        extension_corrupted = group_overlap_check(group_train, group_test)  # policy ignores time input

    temporal_uncorrupted = temporal_integrity_rule(meta["decision_time"].values[train_idx], cutoff)
    temporal_corrupted = temporal_integrity_rule(corrupted_time_train, cutoff)

    return {
        "dataset": dataset, "identifier": identifier_col, "time_aware_policy": time_aware,
        "extension_status_uncorrupted": extension_uncorrupted["status"],
        "extension_fraction_uncorrupted": extension_uncorrupted["overlap_fraction"],
        "extension_status_corrupted_timestamps": extension_corrupted["status"],
        "extension_fraction_corrupted_timestamps": extension_corrupted["overlap_fraction"],
        "temporal_status_uncorrupted": temporal_uncorrupted["status"],
        "temporal_status_corrupted_timestamps": temporal_corrupted["status"],
    }


if __name__ == "__main__":
    rows = [
        run("dataco", lambda ref: case_group_overlap_dataco(ref), "order_id", time_aware=False),
        run("olist", lambda ref: case_group_overlap_olist(ref), "customer_id", time_aware=True),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "timestamp_corruption_check.csv", index=False)
    print(out.to_string(index=False))
    print("\nSaved to", OUT_DIR / "timestamp_corruption_check.csv")
