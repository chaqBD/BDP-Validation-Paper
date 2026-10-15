"""Single canonical source for every release-review number reported in the paper
and the results report. Every table either document cites is copied verbatim from
this script's output, never hand-computed separately. Reports PASS, NEEDS_REVIEW
and BLOCKED as three distinct rates throughout; never collapses NEEDS_REVIEW and
BLOCKED into one "flagged" number without also reporting them separately.

Three outcomes are kept analytically distinct throughout, never conflated into a
single pass/fail judgement:
  (a) ground truth -- whether the case actually has an injected defect
      (`has_defect`, `defect_type`);
  (b) check-specific detection -- whether the one named check relevant to that
      defect type fired (Section 2, the PRIMARY detection endpoint, computed
      identically for every model including Olist random forest); and
  (c) same-period metrics corroboration -- whether the model's own metrics,
      judged purely on frozen AUC/Brier/calibration thresholds with no schema,
      overlap or group information (`verdict_metrics_only`), ALSO flag the
      same case. This is corroboration by a prespecified rule evaluated on the
      same period, not independent confirmation that a release recommendation
      was correct -- both numbers come from the same data. Section 4 uses (c)
      to split every non-defect case's flags into "metrics-corroborated" versus
      "methodology-specific" (flagged by this approach but not corroborated).
      `scripts/run_subsequent_period_check.py` provides the genuinely external
      check (c) cannot: whether a flag on one rolling-origin period predicts a
      subsequent, independently fitted period failing a common acceptance
      criterion.
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
              "verdict_stock_gate", "verdict_extended_gate", "verdict_temporal_integrity"]
APPROACH_LABELS = {
    "verdict_auroc_only": "AUROC-only",
    "verdict_metrics_only": "Metrics-only",
    "verdict_checklist": "Automated checklist",
    "verdict_stock_gate": "Stock gate",
    "verdict_extended_gate": "Extended gate",
    "verdict_temporal_integrity": "Temporal integrity",
}

# The single stock-gate check whose own documentation makes it the one most
# relevant to each defect type -- there is no such stock-gate check for
# group_overlap (that absence is the structural-blind-spot finding itself); kept
# separate from the extension and the temporal check, never combined with them
# into one boolean, so an approach's detection is always attributable to the one
# mechanism that produced it.
STOCK_CHECK_MAP = {
    "leaky_feature": "target_leakage",
    "exact_duplicate": "split_overlap",
    "invalid_schema": "feature_contract",
    "group_overlap": None,
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

    # 2. PRIMARY detection endpoint: a COMPLETE approach x defect x instance table
    #    -- every one of the three check-specific mechanisms (stock gate's own
    #    named check, the group-identifier extension, the temporal-integrity
    #    check) reported as its OWN column, for EVERY defect-model combination,
    #    never combined into a single "relevant_check_fired" boolean. Computed
    #    identically for every model family, including Olist random forest
    #    (known_baseline_issue rows are flagged in their own column rather than
    #    excluded, since this endpoint checks a named check's own flag, not the
    #    aggregate gate verdict RF's pre-existing baseline problem would
    #    otherwise confound).
    print("=" * 100)
    print("2. PRIMARY: complete approach x defect x instance detection table (all models)")
    print("=" * 100)
    rows2p = []
    for _, r in df[df["has_defect"]].iterrows():
        stock_check_name = STOCK_CHECK_MAP.get(r["defect_type"])
        stock_fired = None
        if stock_check_name is not None:
            gj = load_gate_json(r["dataset"], r["case"], r["model"])
            stock_fired = False
            if gj is not None:
                for cat, check_name, flag in per_check_flags(gj):
                    if check_name == stock_check_name and flag not in ("OK", "NOT_APPLICABLE"):
                        stock_fired = True
        extension_fired = r.get("group_overlap_check_status") in ("BLOCKED", "NEEDS_REVIEW")
        temporal_fired = r.get("verdict_temporal_integrity") in ("BLOCKED", "NEEDS_REVIEW")
        rows2p.append({
            "dataset": r["dataset"], "defect_type": r["defect_type"], "model": r["model"],
            "known_baseline_issue": r["known_baseline_issue"],
            "stock_check_name": stock_check_name,
            "stock_named_check_fired": stock_fired,
            "extension_fired": extension_fired,
            "temporal_integrity_fired": temporal_fired,
        })
    primary_detection = pd.DataFrame(rows2p)
    primary_detection.to_csv(OUT_DIR / "primary_defect_check_detection.csv", index=False)
    print(primary_detection.to_string())
    print("\nMean fired rate by dataset x defect_type (each mechanism reported on its own; "
          "NaN = no stock-gate check targets this defect type):")
    print(primary_detection.groupby(["dataset", "defect_type"])[
        ["stock_named_check_fired", "extension_fired", "temporal_integrity_fired"]].mean().to_string())

    # 3. Detection (BLOCKED rate) and review-routing (NEEDS_REVIEW rate) on defect
    #    cases, separately, by defect type x approach -- the aggregate-verdict
    #    view (secondary to Section 2's defect-specific check activation),
    #    known-baseline-issue rows excluded since the aggregate verdict on those
    #    rows is confounded by a pre-existing baseline problem, not the defect
    #    under test (Section 4 reports those rows on their own).
    print("\n" + "=" * 100)
    print("3. Secondary: aggregate-verdict detection (BLOCKED/NEEDS_REVIEW rate) on defect cases, by defect type")
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

    # 4. False alarms on clean/scenario cases (which are NOT defects), by case x
    #    approach -- split into three outcomes, not two: whether the row is
    #    flagged at all (blocked_rate/needs_review_rate, as before), AND, among
    #    flagged rows, whether the metrics-only rule (frozen AUC/Brier/
    #    calibration thresholds, no schema/overlap/group information) ALSO
    #    flags the same row. This is CORROBORATION by a prespecified metrics
    #    rule evaluated on the same period, not independent confirmation that a
    #    release recommendation was actually correct -- both numbers come from
    #    the same data and the same thresholds this paper itself froze, so
    #    agreement only shows internal consistency between two same-period
    #    rules, not external validity. `scripts/run_subsequent_period_check.py`
    #    provides the external check this section cannot: whether a flag on
    #    origin i predicts a genuinely subsequent, independently fitted origin
    #    i+1 failing a common acceptance criterion.
    print("\n" + "=" * 100)
    print("4. False alarms: flag rate on clean/scenario cases, split by same-period metrics corroboration")
    print("=" * 100)
    rows3 = []
    for dataset in df_clean_models.dataset.unique():
        d = df_clean_models[(df_clean_models.dataset == dataset) & (~df_clean_models.has_defect)]
        for case in d.case.unique():
            dd = d[d.case == case]
            metrics_rule_also_flags = dd["verdict_metrics_only"].isin(["BLOCKED", "NEEDS_REVIEW"])
            for appr in APPROACHES:
                flagged = dd[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
                rows3.append({
                    "dataset": dataset, "case": case, "approach": APPROACH_LABELS[appr],
                    "n_cases": len(dd),
                    "blocked_rate": rate(dd[appr], ["BLOCKED"]),
                    "needs_review_rate": rate(dd[appr], ["NEEDS_REVIEW"]),
                    "flagged_rate": flagged.mean() if len(dd) else np.nan,
                    "metrics_corroborated_flag_rate": (flagged & metrics_rule_also_flags).mean() if len(dd) else np.nan,
                    "methodology_specific_flag_rate": (flagged & ~metrics_rule_also_flags).mean() if len(dd) else np.nan,
                })
    false_alarms = pd.DataFrame(rows3)
    false_alarms.to_csv(OUT_DIR / "flagrate_by_nondefect_case.csv", index=False)
    print(false_alarms.pivot_table(index=["dataset", "case"], columns="approach",
                                    values="blocked_rate").to_string())
    print("\n(methodology-specific flag rate = flagged by this approach but NOT by the same-period "
          "metrics-only rule -- a same-period consistency check, not independent validation)")
    print(false_alarms.pivot_table(index=["dataset", "case"], columns="approach",
                                    values="methodology_specific_flag_rate").to_string())

    # 5. Known-baseline-issue rows, reported on their own, every approach's verdict,
    #    not counted anywhere above.
    print("\n" + "=" * 100)
    print("5. Known-baseline-issue rows (Olist random forest) -- defensible blocks, reported separately")
    print("=" * 100)
    kb = df[df["known_baseline_issue"]]
    print(kb[["dataset", "case", "model", "auc", "cal_slope"] + APPROACHES].to_string())

    # 6. Does the extension change the actual release action, or only add diagnostic
    #    detail? Compare stock vs extended verdict per row; report only the rows
    #    where the verdict category actually differs.
    print("\n" + "=" * 100)
    print("6. Stock vs extended gate: rows where the verdict itself changes")
    print("=" * 100)
    order = {"PASS": 0, "NEEDS_REVIEW": 1, "BLOCKED": 2}
    df["stock_rank"] = df["verdict_stock_gate"].map(order)
    df["extended_rank"] = df["verdict_extended_gate"].map(order)
    changed = df[df["stock_rank"] != df["extended_rank"]][
        ["dataset", "case", "model", "has_defect", "known_baseline_issue",
         "verdict_stock_gate", "verdict_extended_gate", "group_overlap_fraction"]]
    changed.to_csv(OUT_DIR / "extension_verdict_changes.csv", index=False)
    print(changed.to_string() if len(changed) else "(no rows where the extended verdict differs from stock)")

    # 7. Review burden (NEEDS_REVIEW + BLOCKED rate) across the whole battery,
    #    known-baseline-issue rows excluded, reported per approach.
    print("\n" + "=" * 100)
    print("7. Review burden (NEEDS_REVIEW or BLOCKED), whole battery, by approach")
    print("=" * 100)
    for appr in APPROACHES:
        print(f"  {APPROACH_LABELS[appr]:22s} "
              f"PASS={rate(df_clean_models[appr], ['PASS']):.2%}  "
              f"NEEDS_REVIEW={rate(df_clean_models[appr], ['NEEDS_REVIEW']):.2%}  "
              f"BLOCKED={rate(df_clean_models[appr], ['BLOCKED']):.2%}")

    # 8. Cost
    print("\n" + "=" * 100)
    print("8. Cost (mean seconds per case)")
    print("=" * 100)
    for col in ["fit_time_s", "time_auroc_only_s", "time_metrics_only_s", "time_checklist_s",
                "time_stock_gate_s", "time_extended_gate_s", "time_temporal_integrity_s"]:
        print(f"  {col:22s} mean={df[col].mean():.4f}  max={df[col].max():.4f}")

    # 9. Check coverage (skipped checks), from the saved gate JSON.
    print("\n" + "=" * 100)
    print("9. Check coverage (stock gate)")
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

    # 10. Supplementary: every individual stock-gate check that fired on every
    #     defect case, all models included (diagnostic detail; Section 2's
    #     defect-specific check activation is the primary detection endpoint).
    print("\n" + "=" * 100)
    print("10. Supplementary: every individual stock-gate check fired on each defect type, all models")
    print("=" * 100)
    rows9 = []
    for _, r in df.iterrows():
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

    # 11. Supplementary evaluation summary (run_heldout_evaluation.py):
    #     severity/seed sweep on exact-duplicate -- reporting the actual
    #     duplication-specific overlap statistic, not only the aggregate verdict
    #     -- and flag rate across eight rolling-origin periods per dataset not
    #     used by either hand-picked scenario case (supplementary, not
    #     genuinely unseen: these periods were already analysed in the
    #     companion study's O4 rolling-origin analysis before this battery
    #     existed), split by same-period metrics corroboration as in Section 4.
    print("\n" + "=" * 100)
    print("11. Supplementary evaluation: severity/seed sweep + eight supplementary periods per dataset")
    print("=" * 100)
    sev_path = OUT_DIR / "heldout_severity_sweep.csv"
    per_path = OUT_DIR / "heldout_period_sweep.csv"
    if sev_path.exists() and per_path.exists():
        sev = pd.read_csv(sev_path)
        per = pd.read_csv(per_path)
        heldout_approaches = ["verdict_checklist", "verdict_stock_gate",
                               "verdict_extended_gate", "verdict_temporal_integrity"]
        for appr in heldout_approaches:
            sev[f"{appr}_detected"] = sev[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
            per[f"{appr}_flagged"] = per[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
        sev["split_overlap_specific_fired"] = sev["split_overlap_flag"] != "OK"
        print("Duplication-SPECIFIC detection (split_overlap check itself) vs. aggregate flag, by severity:")
        print(sev.groupby(["dataset", "severity_frac"])[
            ["split_overlap_specific_fired", "split_overlap_fraction"]
            + [f"{a}_detected" for a in heldout_approaches]].mean().to_string())

        print("\nSupplementary periods: flag rate, split by same-period metrics corroboration:")
        per["metrics_rule_also_flags"] = per["verdict_metrics_only"].isin(["BLOCKED", "NEEDS_REVIEW"])
        rows11 = []
        for dataset in per.dataset.unique():
            dd = per[per.dataset == dataset]
            for appr in heldout_approaches:
                flagged = dd[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
                rows11.append({
                    "dataset": dataset, "approach": APPROACH_LABELS[appr],
                    "flagged_rate": flagged.mean(),
                    "metrics_corroborated_flag_rate": (flagged & dd["metrics_rule_also_flags"]).mean(),
                    "methodology_specific_flag_rate": (flagged & ~dd["metrics_rule_also_flags"]).mean(),
                })
        supp_period_corrob = pd.DataFrame(rows11)
        supp_period_corrob.to_csv(OUT_DIR / "supplementary_period_corroboration.csv", index=False)
        print(supp_period_corrob.to_string(index=False))
        print(f"\n(for reference: the metrics-only rule itself flags "
              f"{per[per.dataset=='olist']['metrics_rule_also_flags'].mean():.0%} of Olist's 8 supplementary "
              f"periods and {per[per.dataset=='dataco']['metrics_rule_also_flags'].mean():.0%} of DataCo's -- "
              "a BLOCKED/NEEDS_REVIEW verdict on one of those periods is not automatically a false alarm)")
    else:
        print("(run scripts/run_heldout_evaluation.py first)")
