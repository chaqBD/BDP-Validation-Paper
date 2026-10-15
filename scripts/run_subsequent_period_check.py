"""Checks every approach's verdict against a genuinely subsequent, untouched
period rather than against a same-period metrics rule.

Agreement between an approach's flag and the metrics-only rule's own flag on the
SAME period (what `analyze_release_review.py` calls "corroboration by a
prespecified metrics rule," Section 4) is not independent evidence that a
release recommendation was actually correct -- both are computed from the same
data. This script instead asks the question a release-review process actually
cares about operationally: when origin i's model was flagged (or not) by a given
approach, did the NEXT origin's independently fitted model (origin i+1, built
from data the approach reviewing origin i never saw) go on to cross a common,
pre-specified acceptance threshold (AUROC >= 0.65, the same frozen threshold
used throughout this paper)?

This reuses the already-computed GBM verdicts for all ten rolling-origin windows
per dataset -- the main battery's two hand-picked scenario cases
(`release_review_battery.csv`) plus the eight supplementary periods
(`heldout_period_sweep.csv`, `run_heldout_evaluation.py`) -- rather than any
separately regenerated analysis, so the next-period AUC is guaranteed to come
from the same current splitter/population code as every other number in this
battery.
"""
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
ACCEPTANCE_AUC = 0.65

# Origin index each dataset's two main-battery scenario cases correspond to
# (fixed in `scripts/run_release_review.py::build_case_list`).
SCENARIO_ORIGIN = {
    ("dataco", "scenario_drift"): 9, ("dataco", "scenario_prevalence"): 1,
    ("olist", "scenario_drift"): 9, ("olist", "scenario_prevalence"): 3,
}


def build_origin_table(dataset):
    battery = pd.read_csv(OUT_DIR / "release_review_battery.csv")
    battery = battery[(battery.dataset == dataset) & (battery.model == "gbm")]
    period_sweep = pd.read_csv(OUT_DIR / "heldout_period_sweep.csv")
    period_sweep = period_sweep[period_sweep.dataset == dataset]

    rows = []
    for (ds, case_name), origin_index in SCENARIO_ORIGIN.items():
        if ds != dataset:
            continue
        r = battery[battery.case == case_name]
        if not len(r):
            continue
        r = r.iloc[0]
        rows.append({"origin_index": origin_index, "auc": r["auc"],
                     **{a: r[a] for a in APPROACHES}})
    for _, r in period_sweep.iterrows():
        rows.append({"origin_index": r["origin_index"], "auc": r["auc"],
                     **{a: r[a] for a in APPROACHES}})
    origins = pd.DataFrame(rows).sort_values("origin_index").reset_index(drop=True)
    origins["next_auc"] = origins["auc"].shift(-1)
    origins["next_origin_fails_acceptance"] = origins["next_auc"] < ACCEPTANCE_AUC
    return origins.iloc[:-1]  # drop the last origin: it has no next period


if __name__ == "__main__":
    all_rows = []
    for dataset in ["dataco", "olist"]:
        origins = build_origin_table(dataset)
        origins["dataset"] = dataset
        all_rows.append(origins)
        print(f"\n{dataset} -- {len(origins)} origins with a usable next period")
        print(origins[["origin_index", "auc", "next_auc", "next_origin_fails_acceptance"]]
              .to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_DIR / "subsequent_period_check.csv", index=False)
    print("\nSaved to", OUT_DIR / "subsequent_period_check.csv")

    print("\n" + "=" * 100)
    print("Does a flag on origin i predict the NEXT origin's own model failing the "
          f"{ACCEPTANCE_AUC} AUROC acceptance criterion? (n={len(combined)} origin-pairs, both datasets pooled)")
    print("=" * 100)
    rows = []
    for appr in APPROACHES:
        flagged = combined[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
        fails_next = combined["next_origin_fails_acceptance"]
        tp = int((flagged & fails_next).sum())
        fp = int((flagged & ~fails_next).sum())
        fn = int((~flagged & fails_next).sum())
        tn = int((~flagged & ~fails_next).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        rows.append({"approach": appr, "flagged_n": int(flagged.sum()), "tp": tp, "fp": fp,
                     "fn": fn, "tn": tn, "precision": precision, "recall": recall})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "subsequent_period_check_summary.csv", index=False)
    print(result.to_string(index=False))
    print("\n(tp = flagged this origin AND next origin's own model failed the acceptance "
          "criterion; fp = flagged but next origin passed; precision/recall are with "
          "respect to predicting next-period acceptance failure, pooled across both "
          "datasets' 9 usable origin-pairs each -- too few origin-pairs for a formal "
          "significance test, reported descriptively, Section 7.)")
