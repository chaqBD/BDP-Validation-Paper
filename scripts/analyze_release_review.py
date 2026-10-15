"""Post-process the release-review battery: compute detection sensitivity, false-alarm
rate, review burden, skipped-check rate and cost per comparator approach, plus the
check-by-scenario detection matrix for the stock gate's own checks.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "release_review"


def load_gate_json(dataset, case, model):
    path = OUT_DIR / "gate_json" / dataset / f"{dataset}_{case}_{model}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flag_rate(df, col, flagged_values=("NEEDS_REVIEW", "BLOCKED")):
    return df[col].isin(flagged_values).mean()


def per_check_flags(gjson):
    """Returns list of (category, check_name, flag) for every check in a gate report."""
    out = []
    cats = gjson.get("results_by_category", {})
    for cat, items in cats.items():
        for it in items:
            out.append((cat, it.get("check_name"), it.get("flag")))
    return out


if __name__ == "__main__":
    df = pd.read_csv(OUT_DIR / "release_review_battery.csv")
    approaches = ["verdict_auroc_only", "verdict_checklist", "verdict_stock_gate",
                  "verdict_extended_gate"]

    # 1. Detection sensitivity and false-alarm rate per approach, per dataset
    rows = []
    for dataset in df.dataset.unique():
        d = df[df.dataset == dataset]
        defects = d[d.has_defect]
        clean_scenario = d[~d.has_defect]
        for appr in approaches:
            rows.append({
                "dataset": dataset, "approach": appr,
                "detection_sensitivity": flag_rate(defects, appr),
                "false_alarm_rate": flag_rate(clean_scenario, appr),
                "review_burden_overall": flag_rate(d, appr),
                "unsupported_approval_rate": 1 - flag_rate(defects, appr),
                "mean_time_s": d[appr.replace("verdict_", "time_") + "_s"].mean()
                if (appr.replace("verdict_", "time_") + "_s") in d.columns else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "approach_summary.csv", index=False)
    print("Approach summary (detection sensitivity, false alarms, review burden):")
    print(summary.to_string())

    # 2. Detection sensitivity broken out by defect type x approach
    rows2 = []
    for dataset in df.dataset.unique():
        d = df[(df.dataset == dataset) & (df.has_defect)]
        for defect_type in d.defect_type.unique():
            dd = d[d.defect_type == defect_type]
            for appr in approaches:
                rows2.append({"dataset": dataset, "defect_type": defect_type,
                               "approach": appr, "detection_rate": flag_rate(dd, appr),
                               "n_cases": len(dd)})
    by_defect = pd.DataFrame(rows2)
    by_defect.to_csv(OUT_DIR / "detection_by_defect_type.csv", index=False)
    print("\nDetection rate by defect type x approach:")
    print(by_defect.pivot_table(index=["dataset", "defect_type"], columns="approach",
                                 values="detection_rate").to_string())

    # 3. Scenario-case breakdown (false alarms specifically, since these should NOT be flagged)
    rows3 = []
    for dataset in df.dataset.unique():
        d = df[(df.dataset == dataset) & (~df.has_defect)]
        for case in d.case.unique():
            dd = d[d.case == case]
            for appr in approaches:
                rows3.append({"dataset": dataset, "case": case, "approach": appr,
                               "flag_rate": flag_rate(dd, appr), "n_cases": len(dd)})
    by_nondefect = pd.DataFrame(rows3)
    by_nondefect.to_csv(OUT_DIR / "flagrate_by_nondefect_case.csv", index=False)
    print("\nFlag rate on clean/scenario cases (should be low -- these are not defects):")
    print(by_nondefect.pivot_table(index=["dataset", "case"], columns="approach",
                                    values="flag_rate").to_string())

    # 4. Skipped-check and per-check flag matrix for the stock gate, from the saved JSON
    rows4 = []
    for _, r in df.iterrows():
        gj = load_gate_json(r["dataset"], r["case"], r["model"])
        if gj is None:
            continue
        checks = per_check_flags(gj)
        n_total = len(checks)
        n_skipped = sum(1 for _, _, f in checks if f == "NOT_APPLICABLE")
        n_flagged = sum(1 for _, _, f in checks if f not in ("OK", "NOT_APPLICABLE"))
        rows4.append({"dataset": r["dataset"], "case": r["case"], "model": r["model"],
                      "has_defect": r["has_defect"], "n_checks_total": n_total,
                      "n_checks_skipped": n_skipped, "n_checks_flagged": n_flagged,
                      "skip_rate": n_skipped / n_total if n_total else np.nan})
    coverage = pd.DataFrame(rows4)
    coverage.to_csv(OUT_DIR / "check_coverage.csv", index=False)
    print("\nCheck coverage (mean skip rate by case):")
    print(coverage.groupby(["dataset", "case"])["skip_rate"].mean().to_string())

    # 5. Which specific check names fired on which defect type (detection matrix)
    rows5 = []
    for _, r in df.iterrows():
        if not r["has_defect"]:
            continue
        gj = load_gate_json(r["dataset"], r["case"], r["model"])
        if gj is None:
            continue
        for cat, name, flag in per_check_flags(gj):
            if flag not in ("OK", "NOT_APPLICABLE"):
                rows5.append({"dataset": r["dataset"], "defect_type": r["defect_type"],
                              "model": r["model"], "category": cat, "check_name": name,
                              "flag": flag})
    fired = pd.DataFrame(rows5)
    fired.to_csv(OUT_DIR / "checks_fired_on_defects.csv", index=False)
    print("\nWhich checks fired on which defect types (stock gate):")
    if len(fired):
        print(fired.groupby(["defect_type", "check_name"]).size().to_string())
