"""Single canonical source for every release-review number reported in the paper
and the results report. Every table either document cites is copied verbatim from
this script's output, never hand-computed separately. Reports PASS, NEEDS_REVIEW
and BLOCKED as three distinct rates throughout; never collapses NEEDS_REVIEW and
BLOCKED into one "flagged" number without also reporting them separately.
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
APPROACHES = ["verdict_auroc_only", "verdict_metrics_only", "verdict_checklist",
              "verdict_stock_gate", "verdict_extended_gate"]
APPROACH_LABELS = {
    "verdict_auroc_only": "AUROC-only",
    "verdict_metrics_only": "Metrics-only",
    "verdict_checklist": "Automated checklist",
    "verdict_stock_gate": "Stock gate",
    "verdict_extended_gate": "Extended gate",
}


def rate(series, values):
    return series.isin(values).mean() if len(series) else np.nan


def load_gate_json(dataset, case, model):
    path = OUT_DIR / "gate_json" / dataset / f"{dataset}_{case}_{model}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_check_flags(gjson):
    out = []
    for cat, items in gjson.get("results_by_category", {}).items():
        for it in items:
            out.append((cat, it.get("check_name"), it.get("flag")))
    return out


if __name__ == "__main__":
    df = pd.read_csv(OUT_DIR / "release_review_battery.csv")
    df_clean_models = df[~df["known_baseline_issue"]]  # excludes (olist, rf) everywhere

    # 1. PASS / NEEDS_REVIEW / BLOCKED breakdown, every approach, every case,
    #    known-baseline-issue rows excluded (reported separately in section 5).
    print("=" * 100)
    print("1. Verdict breakdown by case x approach (known-baseline-issue rows excluded)")
    print("=" * 100)
    rows1 = []
    for dataset in df_clean_models.dataset.unique():
        d = df_clean_models[df_clean_models.dataset == dataset]
        for case in d.case.unique():
            dd = d[d.case == case]
            row = {"dataset": dataset, "case": case, "has_defect": bool(dd.has_defect.iloc[0]),
                   "n": len(dd)}
            for appr in APPROACHES:
                for v in ["PASS", "NEEDS_REVIEW", "BLOCKED"]:
                    row[f"{appr}_{v}"] = int((dd[appr] == v).sum())
            rows1.append(row)
    verdict_breakdown = pd.DataFrame(rows1)
    verdict_breakdown.to_csv(OUT_DIR / "verdict_breakdown.csv", index=False)
    print(verdict_breakdown.to_string())

    # 2. Detection (BLOCKED rate) and review-routing (NEEDS_REVIEW rate) on defect
    #    cases, separately, by defect type x approach.
    print("\n" + "=" * 100)
    print("2. Detection: BLOCKED rate and NEEDS_REVIEW rate on defect cases, by defect type")
    print("=" * 100)
    rows2 = []
    for dataset in df_clean_models.dataset.unique():
        d = df_clean_models[(df_clean_models.dataset == dataset) & (df_clean_models.has_defect)]
        for defect_type in d.defect_type.unique():
            dd = d[d.defect_type == defect_type]
            for appr in APPROACHES:
                rows2.append({
                    "dataset": dataset, "defect_type": defect_type, "approach": APPROACH_LABELS[appr],
                    "n_cases": len(dd),
                    "blocked_rate": rate(dd[appr], ["BLOCKED"]),
                    "needs_review_rate": rate(dd[appr], ["NEEDS_REVIEW"]),
                    "pass_rate_unsupported_approval": rate(dd[appr], ["PASS"]),
                })
    detection = pd.DataFrame(rows2)
    detection.to_csv(OUT_DIR / "detection_by_defect_type.csv", index=False)
    print(detection.pivot_table(index=["dataset", "defect_type"], columns="approach",
                                 values="blocked_rate").to_string())
    print("\n(unsupported-approval rate = PASS rate on a true-defect case, NOT the complement of the block rate)")
    print(detection.pivot_table(index=["dataset", "defect_type"], columns="approach",
                                 values="pass_rate_unsupported_approval").to_string())

    # 3. False alarms on clean/scenario cases (which are NOT defects), by case x approach.
    print("\n" + "=" * 100)
    print("3. False alarms: BLOCKED rate and NEEDS_REVIEW rate on clean/scenario cases")
    print("=" * 100)
    rows3 = []
    for dataset in df_clean_models.dataset.unique():
        d = df_clean_models[(df_clean_models.dataset == dataset) & (~df_clean_models.has_defect)]
        for case in d.case.unique():
            dd = d[d.case == case]
            for appr in APPROACHES:
                rows3.append({
                    "dataset": dataset, "case": case, "approach": APPROACH_LABELS[appr],
                    "n_cases": len(dd),
                    "blocked_rate": rate(dd[appr], ["BLOCKED"]),
                    "needs_review_rate": rate(dd[appr], ["NEEDS_REVIEW"]),
                })
    false_alarms = pd.DataFrame(rows3)
    false_alarms.to_csv(OUT_DIR / "flagrate_by_nondefect_case.csv", index=False)
    print(false_alarms.pivot_table(index=["dataset", "case"], columns="approach",
                                    values="blocked_rate").to_string())

    # 4. Known-baseline-issue rows, reported on their own, every approach's verdict,
    #    not counted anywhere above.
    print("\n" + "=" * 100)
    print("4. Known-baseline-issue rows (Olist random forest) -- defensible blocks, reported separately")
    print("=" * 100)
    kb = df[df["known_baseline_issue"]]
    print(kb[["dataset", "case", "model", "auc", "cal_slope"] + APPROACHES].to_string())

    # 5. Does the extension change the actual release action, or only add diagnostic
    #    detail? Compare stock vs extended verdict per row; report only the rows
    #    where the verdict category actually differs.
    print("\n" + "=" * 100)
    print("5. Stock vs extended gate: rows where the verdict itself changes")
    print("=" * 100)
    order = {"PASS": 0, "NEEDS_REVIEW": 1, "BLOCKED": 2}
    df["stock_rank"] = df["verdict_stock_gate"].map(order)
    df["extended_rank"] = df["verdict_extended_gate"].map(order)
    changed = df[df["stock_rank"] != df["extended_rank"]][
        ["dataset", "case", "model", "has_defect", "known_baseline_issue",
         "verdict_stock_gate", "verdict_extended_gate", "group_overlap_fraction"]]
    changed.to_csv(OUT_DIR / "extension_verdict_changes.csv", index=False)
    print(changed.to_string() if len(changed) else "(no rows where the extended verdict differs from stock)")

    # 6. Review burden (NEEDS_REVIEW + BLOCKED rate) across the whole battery,
    #    known-baseline-issue rows excluded, reported per approach.
    print("\n" + "=" * 100)
    print("6. Review burden (NEEDS_REVIEW or BLOCKED), whole battery, by approach")
    print("=" * 100)
    for appr in APPROACHES:
        print(f"  {APPROACH_LABELS[appr]:22s} "
              f"PASS={rate(df_clean_models[appr], ['PASS']):.2%}  "
              f"NEEDS_REVIEW={rate(df_clean_models[appr], ['NEEDS_REVIEW']):.2%}  "
              f"BLOCKED={rate(df_clean_models[appr], ['BLOCKED']):.2%}")

    # 7. Cost
    print("\n" + "=" * 100)
    print("7. Cost (mean seconds per case)")
    print("=" * 100)
    for col in ["fit_time_s", "time_auroc_only_s", "time_metrics_only_s", "time_checklist_s",
                "time_stock_gate_s", "time_extended_gate_s"]:
        print(f"  {col:22s} mean={df[col].mean():.4f}  max={df[col].max():.4f}")

    # 8. Check coverage (skipped checks), from the saved gate JSON.
    print("\n" + "=" * 100)
    print("8. Check coverage (stock gate)")
    print("=" * 100)
    rows8 = []
    for _, r in df.iterrows():
        gj = load_gate_json(r["dataset"], r["case"], r["model"])
        if gj is None:
            continue
        checks = per_check_flags(gj)
        n_total = len(checks)
        n_skipped = sum(1 for _, _, f in checks if f == "NOT_APPLICABLE")
        n_flagged = sum(1 for _, _, f in checks if f not in ("OK", "NOT_APPLICABLE"))
        rows8.append({"dataset": r["dataset"], "case": r["case"], "model": r["model"],
                      "n_checks_total": n_total, "n_checks_skipped": n_skipped,
                      "n_checks_flagged": n_flagged,
                      "skip_rate": n_skipped / n_total if n_total else np.nan})
    coverage = pd.DataFrame(rows8)
    coverage.to_csv(OUT_DIR / "check_coverage.csv", index=False)
    print(coverage.groupby(["dataset", "case"])["skip_rate"].mean().to_string())
    print(f"\n  overall mean n_checks_total={coverage.n_checks_total.mean():.1f}, "
          f"mean skip_rate={coverage.skip_rate.mean():.3f}")

    # 9. Which individual stock-gate check fired on which defect type.
    print("\n" + "=" * 100)
    print("9. Which individual stock-gate check fired on which defect type")
    print("=" * 100)
    rows9 = []
    for _, r in df_clean_models.iterrows():
        if not r["has_defect"]:
            continue
        gj = load_gate_json(r["dataset"], r["case"], r["model"])
        if gj is None:
            continue
        for cat, name, flag in per_check_flags(gj):
            if flag not in ("OK", "NOT_APPLICABLE"):
                rows9.append({"dataset": r["dataset"], "defect_type": r["defect_type"],
                              "model": r["model"], "check_name": name, "flag": flag})
    fired = pd.DataFrame(rows9)
    fired.to_csv(OUT_DIR / "checks_fired_on_defects.csv", index=False)
    if len(fired):
        print(fired.groupby(["dataset", "defect_type", "check_name"]).size().to_string())
