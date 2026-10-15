"""Paper B main battery: for each dataset, build the clean reference, four defect
cases and two scenario cases; for each, fit three frozen model families; score with
six comparator approaches (AUROC-only, metrics-only, automated checklist, stock
gate, extended gate, temporal integrity); save one frozen results table with
ground-truth labels and every approach's verdict (PASS / NEEDS_REVIEW / BLOCKED kept
separate throughout, never collapsed inside this script).

The group-identifier policy for the extended check is declared once, per dataset,
before any case is built, and applied identically to every case: DataCo checks
order-id overlap (no time-awareness needed -- an order has one decision time, so any
split of its rows between train and test is illegitimate regardless of when); Olist
checks customer-id overlap, time-aware (a customer's genuinely past orders in
training do not count, only same-window-or-later ones do). No branch in this file
selects the identifier or the policy based on which case is being scored.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.gate_integration import run_gate_report
from src.metrics import all_metrics
from src.release_cases import (_reference, case_clean, case_leaky_feature,
                                case_exact_duplicate, case_group_overlap_dataco,
                                case_group_overlap_olist, case_invalid_schema,
                                case_scenario)
from src.release_review import (MODEL_FAMILIES, fit_predict, auroc_only_rule,
                                 metrics_only_rule, automated_checklist_rule,
                                 group_overlap_check, temporal_integrity_rule,
                                 combine_verdicts, KNOWN_BASELINE_ISSUE)

OUT_DIR = RESULTS_DIR / "release_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Declared once, in advance, per dataset -- never selected per case.
GROUP_POLICY = {
    "dataco": {"identifier": "order_id", "time_aware": False},
    "olist": {"identifier": "customer_id", "time_aware": True},
}

# Cases that are themselves defined by row-level duplication: the "no row is ever
# both trained on and scored" integrity assertion legitimately does not apply to
# these, since exact reuse is the ground truth being tested.
EXPECTED_ROW_OVERLAP_CASES = {"defect_exact_duplicate"}


def build_case_list(dataset):
    ref = _reference(dataset)
    cases = [
        ("clean_reference", "clean", False, case_clean(ref)),
        ("defect_leaky_feature", "leaky_feature", True, case_leaky_feature(ref, dataset)),
        ("defect_exact_duplicate", "exact_duplicate", True, case_exact_duplicate(ref)),
        ("defect_invalid_schema", "invalid_schema", True, case_invalid_schema(ref)),
    ]
    if dataset == "dataco":
        cases.append(("defect_group_overlap", "group_overlap", True,
                      case_group_overlap_dataco(ref)))
    else:
        cases.append(("defect_group_overlap", "group_overlap", True,
                      case_group_overlap_olist(ref)))

    if dataset == "dataco":
        drift = case_scenario("dataco", origin_index=9)
        prevalence = case_scenario("dataco", origin_index=1)
    else:
        drift = case_scenario("olist", origin_index=9)
        prevalence = case_scenario("olist", origin_index=3)
    cases.append(("scenario_drift", "scenario_drift", False, drift))
    cases.append(("scenario_prevalence", "scenario_prevalence", False, prevalence))
    return ref, cases


def run_one_case(dataset, case_name, defect_type, has_defect, case, ref, family,
                  cap_rows=4000):
    X = case.get("X", ref["X"])
    y = case.get("y", ref["y"])
    meta = case.get("meta", ref["meta"])
    num = case.get("num_override", case.get("num", ref["num"]))
    cat = case.get("cat", ref["cat"])
    train_idx, test_idx = case["train_idx"], case["test_idx"]

    row_overlap = np.intersect1d(train_idx, test_idx)
    if case_name in EXPECTED_ROW_OVERLAP_CASES:
        assert len(row_overlap) > 0, f"{case_name} is defined by row reuse but found none"
    else:
        assert len(row_overlap) == 0, (
            f"{dataset}/{case_name}: {len(row_overlap)} row(s) are both trained on and "
            f"scored, which is only expected for {EXPECTED_ROW_OVERLAP_CASES}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    t0 = time.time()
    pipe, p_test = fit_predict(family, X_train, y_train, X_test, num, cat)
    fit_time = time.time() - t0
    m = all_metrics(y_test.values, p_test)

    # 1. AUROC-only
    t0 = time.time()
    v_auroc = auroc_only_rule(m["auc"])
    t_auroc = time.time() - t0

    # 2. Metrics-only (previously mislabelled "checklist")
    t0 = time.time()
    v_metrics_only = metrics_only_rule(m["auc"], m["brier"], m["cal_slope"])
    t_metrics_only = time.time() - t0

    # Schema perturbation (invalid-schema case only) applies identically to every
    # review-time consumer of X_test: the stock gate and the automated checklist
    # both see it, since both are meant to validate review-time data integrity.
    X_test_for_review = X_test.copy()
    if case.get("drop_col_for_review"):
        X_test_for_review = X_test_for_review.drop(columns=[case["drop_col_for_review"]])

    # 3. Automated checklist: metrics + a real schema check + a real (hashed, exact)
    # row-overlap check -- distinct from, and weaker than, the stock gate's bootstrap
    # checks, representing a team that writes its own checks without adopting a
    # dedicated governance library.
    t0 = time.time()
    v_checklist = automated_checklist_rule(m["auc"], m["brier"], m["cal_slope"],
                                            X_train, X_test_for_review)
    t_checklist = time.time() - t0

    # 4. Stock gate (model card built internally by run_gate_report from its own
    # template, merged with the dataset name and validation strategy declared here)
    t0 = time.time()
    tag = f"{dataset}_{case_name}_{family}"
    gres = run_gate_report(pipe, X_train, X_test_for_review, y_test.values, p_test,
                            OUT_DIR / "gate_json" / dataset, tag,
                            dataset_name=dataset, validation_strategy="out_of_time")
    t_stock = time.time() - t0
    v_stock = gres.get("gate_status", "ERROR")

    # 5. Extended gate = stock gate + the declared, fixed group-overlap check.
    policy = GROUP_POLICY[dataset]
    group_train = meta[policy["identifier"]].values[train_idx]
    group_test = meta[policy["identifier"]].values[test_idx]
    test_period_start = meta["decision_time"].values[test_idx].min()
    time_train = meta["decision_time"].values[train_idx]
    t0 = time.time()
    if policy["time_aware"]:
        goc = group_overlap_check(group_train, group_test, time_train=time_train,
                                   test_period_start=test_period_start)
    else:
        goc = group_overlap_check(group_train, group_test)
    v_extended = combine_verdicts(v_stock, goc["status"])
    t_extended = time.time() - t0 + t_stock

    # 6. Temporal integrity: a minimal, identifier-free comparator (no group id
    # needed), included to establish what detection a basic chronological check
    # alone already buys, before crediting the group-identifier extension with it.
    t0 = time.time()
    tic = temporal_integrity_rule(time_train, test_period_start)
    v_temporal = tic["status"]
    t_temporal = time.time() - t0

    return {
        "dataset": dataset, "case": case_name, "defect_type": defect_type,
        "has_defect": has_defect, "model": family,
        "known_baseline_issue": (dataset, family) in KNOWN_BASELINE_ISSUE,
        "auc": m["auc"], "brier": m["brier"], "cal_slope": m["cal_slope"], "ece": m["ece"],
        "n_train": len(train_idx), "n_test": len(test_idx), "fit_time_s": fit_time,
        "verdict_auroc_only": v_auroc, "time_auroc_only_s": t_auroc,
        "verdict_metrics_only": v_metrics_only, "time_metrics_only_s": t_metrics_only,
        "verdict_checklist": v_checklist, "time_checklist_s": t_checklist,
        "verdict_stock_gate": v_stock, "time_stock_gate_s": t_stock,
        "verdict_extended_gate": v_extended, "time_extended_gate_s": t_extended,
        "verdict_temporal_integrity": v_temporal, "time_temporal_integrity_s": t_temporal,
        "temporal_integrity_n_violations": tic["n_violations"],
        "group_overlap_identifier": policy["identifier"],
        "group_overlap_fraction": goc["overlap_fraction"],
        "group_overlap_check_status": goc["status"],
    }


if __name__ == "__main__":
    all_rows = []
    for dataset in ["dataco", "olist"]:
        t0 = time.time()
        ref, cases = build_case_list(dataset)
        for case_name, defect_type, has_defect, case in cases:
            for family in MODEL_FAMILIES:
                row = run_one_case(dataset, case_name, defect_type, has_defect, case,
                                    ref, family)
                all_rows.append(row)
                print(f"[{dataset}] {case_name:25s} {family:8s} auc={row['auc']:.3f} "
                      f"auroc_only={row['verdict_auroc_only']:13s} "
                      f"metrics_only={row['verdict_metrics_only']:13s} "
                      f"checklist={row['verdict_checklist']:13s} "
                      f"stock={row['verdict_stock_gate']:13s} "
                      f"extended={row['verdict_extended_gate']:13s} "
                      f"temporal={row['verdict_temporal_integrity']:13s}")
        print(f"[{dataset}] battery done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "release_review_battery.csv", index=False)
    print("\nSaved to", OUT_DIR / "release_review_battery.csv")
