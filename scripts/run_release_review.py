"""Paper B main battery: for each dataset, build the clean reference, four defect
cases and two scenario cases; for each, fit three frozen model families; score with
four comparator approaches (AUROC-only, checklist, stock gate, extended gate); save
one frozen results table with ground-truth labels and every approach's verdict.
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
from src.gate_integration import run_gate_report, GATE_CONFIG, MODEL_CARD_TEMPLATE
from src.metrics import all_metrics
from src.release_cases import (_reference, case_clean, case_leaky_feature,
                                case_exact_duplicate, case_group_overlap_dataco,
                                case_group_overlap_olist, case_invalid_schema,
                                case_scenario)
from src.release_review import (MODEL_FAMILIES, fit_predict, auroc_only_rule,
                                 checklist_rule, group_overlap_check, combine_verdicts)
from bdp_model_gate import StructuredGateContext, ModelGate
from bdp_model_gate.structured import default_structured_checks

OUT_DIR = RESULTS_DIR / "release_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    # Each case may carry its own X/y/meta/num/cat (scenario cases do); defect cases
    # reuse the reference's, with overrides for feature set / dropped column.
    X = case.get("X", ref["X"])
    y = case.get("y", ref["y"])
    meta = case.get("meta", ref["meta"])
    num = case.get("num_override", case.get("num", ref["num"]))
    cat = case.get("cat", ref["cat"])
    train_idx, test_idx = case["train_idx"], case["test_idx"]

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

    # 2. Checklist
    t0 = time.time()
    v_checklist = checklist_rule(m["auc"], m["brier"], m["cal_slope"])
    t_checklist = time.time() - t0

    # 3. Stock gate
    group_col = "order_id" if defect_type != "group_overlap" or dataset == "dataco" else "customer_id"
    model_card = dict(MODEL_CARD_TEMPLATE, training_data_source=dataset,
                       validation_strategy="out_of_time")
    X_test_for_gate = X_test.copy()
    if case.get("drop_col_for_review"):
        X_test_for_gate = X_test_for_gate.drop(columns=[case["drop_col_for_review"]])

    t0 = time.time()
    tag = f"{dataset}_{case_name}_{family}"
    gres = run_gate_report(pipe, X_train, X_test_for_gate, y_test.values, p_test,
                            OUT_DIR / "gate_json" / dataset, tag,
                            dataset_name=dataset, validation_strategy="out_of_time")
    t_stock = time.time() - t0
    v_stock = gres.get("gate_status", "ERROR")

    # 4. Extended gate = stock gate + explicit group-overlap check on the declared group id
    t0 = time.time()
    group_train = meta[group_col].values[train_idx]
    group_test = meta[group_col].values[test_idx]
    goc = group_overlap_check(group_train, group_test)
    v_extended = combine_verdicts(v_stock, goc["status"])
    t_extended = time.time() - t0 + t_stock

    return {
        "dataset": dataset, "case": case_name, "defect_type": defect_type,
        "has_defect": has_defect, "model": family,
        "auc": m["auc"], "brier": m["brier"], "cal_slope": m["cal_slope"], "ece": m["ece"],
        "n_train": len(train_idx), "n_test": len(test_idx), "fit_time_s": fit_time,
        "verdict_auroc_only": v_auroc, "time_auroc_only_s": t_auroc,
        "verdict_checklist": v_checklist, "time_checklist_s": t_checklist,
        "verdict_stock_gate": v_stock, "time_stock_gate_s": t_stock,
        "verdict_extended_gate": v_extended, "time_extended_gate_s": t_extended,
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
                      f"checklist={row['verdict_checklist']:13s} "
                      f"stock={row['verdict_stock_gate']:13s} "
                      f"extended={row['verdict_extended_gate']:13s}")
        print(f"[{dataset}] battery done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "release_review_battery.csv", index=False)
    print("\nSaved to", OUT_DIR / "release_review_battery.csv")
