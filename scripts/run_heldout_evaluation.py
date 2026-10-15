"""Supplementary evaluation of the group-overlap extension and the
temporal-integrity comparator, beyond the single constructed instance per defect
type used in the main battery (`run_release_review.py`). Two independent
extensions of that battery:

1. Severity/seed sweep on the exact-duplicate defect: the only defect whose
   severity is a free parameter (the fraction of test rows also duplicated into
   training) and whose instances can be redrawn under independent seeds. Five
   severities x three seeds each, never seen by any threshold tuned in this study
   (`src.release_review`'s frozen thresholds predate this sweep). Each instance
   gets its own tag (`severity_<frac>_seed_<seed>`) and its own gate-report path,
   so no instance's saved report can overwrite another's or the main battery's --
   an earlier version of this script built every instance under the literal tag
   `defect_exact_duplicate`, so all fifteen per-dataset runs, and potentially the
   main battery's own report for that case, shared one file on disk. For every
   instance this version also extracts the gate's own `split_overlap` sub-check
   (flag and fraction) and the exact injected-row recovery count, not just the
   aggregate PASS/NEEDS_REVIEW/BLOCKED verdict, so a severity sweep that reports
   "100% detection" cannot be mistaken for "the duplication-specific check fired
   100% of the time" (it frequently does not -- see `analyze_release_review.py`).
2. Eight rolling-origin periods per dataset not used by either hand-picked
   scenario case in the main battery. These are **supplementary** periods, not
   genuinely unseen ones: their drift (PSI) and next-period AUC change were
   already computed and reported in the companion study's O4 analysis before this
   battery existed, so "held-out" here means "not used to build this battery's
   two scenario cases," not "never previously examined."

Uses gbm only (the frozen-threshold reference family) to keep the sweep's cost
proportionate to what it adds; the main battery already covers all three model
families on the single hand-picked instance per case.
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
from src.release_cases import _reference, case_exact_duplicate, case_scenario
from scripts.run_release_review import run_one_case
from scripts.run_sampling_sensitivity import extract_split_overlap_json_path

OUT_DIR = RESULTS_DIR / "release_review"
FAMILY = "gbm"

SEVERITIES = [0.02, 0.05, 0.1, 0.2, 0.3]
SEEDS = [101, 202, 303]

# The two origin indices already used as the main battery's hand-picked scenario
# cases, per dataset -- excluded here so every row below is a period not used to
# build either scenario case (not "never previously examined": see module docstring).
MAIN_BATTERY_ORIGINS = {"dataco": {9, 1}, "olist": {9, 3}}
N_ORIGINS = 10


def _gate_json_path(dataset, case_name, family):
    return OUT_DIR / "gate_json" / dataset / f"{dataset}_{case_name}_{family}.json"


if __name__ == "__main__":
    severity_rows = []
    period_rows = []
    manifest_rows = []

    for dataset in ["dataco", "olist"]:
        ref = _reference(dataset)

        # 1. Severity/seed sweep on exact-duplicate, unique tag per instance.
        t0 = time.time()
        for frac in SEVERITIES:
            for seed in SEEDS:
                case = case_exact_duplicate(ref, frac=frac, seed=seed)
                instance_case_name = f"severity_{frac}_seed_{seed}"
                row = run_one_case(dataset, instance_case_name, "exact_duplicate",
                                    True, case, ref, FAMILY)
                row["severity_frac"] = frac
                row["seed"] = seed
                gjson_path = _gate_json_path(dataset, instance_case_name, FAMILY)
                overlap = extract_split_overlap_json_path(gjson_path)
                row["split_overlap_flag"] = overlap.get("overlap_with_training", {}).get("flag")
                row["split_overlap_fraction"] = \
                    overlap.get("overlap_with_training", {}).get("metadata", {}).get("overlap_fraction")
                row["split_overlap_n"] = \
                    overlap.get("overlap_with_training", {}).get("metadata", {}).get("n_overlapping")
                injected_test_rows = set(case.get("injected_test_rows", np.array([], dtype=int)).tolist())
                row["n_injected"] = len(injected_test_rows)
                row["gate_json_path"] = str(gjson_path)
                severity_rows.append(row)
                manifest_rows.append({"sweep": "severity", "dataset": dataset,
                                       "severity_frac": frac, "seed": seed,
                                       "case_name": instance_case_name,
                                       "gate_json_path": str(gjson_path)})
        print(f"[{dataset}] severity/seed sweep done in {time.time()-t0:.1f}s "
              f"({len(SEVERITIES)*len(SEEDS)} instances)")

        # 2. Supplementary rolling-origin periods not used by the main battery
        #    (unique tags already, by origin index -- no collision risk here).
        t0 = time.time()
        held_out_origins = [i for i in range(N_ORIGINS) if i not in MAIN_BATTERY_ORIGINS[dataset]]
        for origin_index in held_out_origins:
            case = case_scenario(dataset, origin_index=origin_index)
            instance_case_name = f"supplementary_period_{origin_index}"
            row = run_one_case(dataset, instance_case_name, "scenario_supplementary",
                                False, case, ref, FAMILY)
            row["origin_index"] = origin_index
            gjson_path = _gate_json_path(dataset, instance_case_name, FAMILY)
            row["gate_json_path"] = str(gjson_path)
            period_rows.append(row)
            manifest_rows.append({"sweep": "period", "dataset": dataset,
                                   "origin_index": origin_index, "case_name": instance_case_name,
                                   "gate_json_path": str(gjson_path)})
        print(f"[{dataset}] supplementary period sweep done in {time.time()-t0:.1f}s "
              f"({len(held_out_origins)} periods)")

    severity_df = pd.DataFrame(severity_rows)
    period_df = pd.DataFrame(period_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    severity_df.to_csv(OUT_DIR / "heldout_severity_sweep.csv", index=False)
    period_df.to_csv(OUT_DIR / "heldout_period_sweep.csv", index=False)
    manifest_df.to_csv(OUT_DIR / "heldout_manifest.csv", index=False)
    print("\nSaved to", OUT_DIR / "heldout_severity_sweep.csv")
    print("Saved to", OUT_DIR / "heldout_period_sweep.csv")
    print("Saved to", OUT_DIR / "heldout_manifest.csv")

    print("\n" + "=" * 100)
    print("Severity sweep: split_overlap-SPECIFIC detection (flag != OK) vs. aggregate flag, by severity")
    print("=" * 100)
    severity_df["split_overlap_specific_fired"] = severity_df["split_overlap_flag"] != "OK"
    for appr in ["verdict_checklist", "verdict_stock_gate", "verdict_extended_gate",
                 "verdict_temporal_integrity"]:
        severity_df[f"{appr}_detected"] = severity_df[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
    print(severity_df.groupby(["dataset", "severity_frac"])[
        ["split_overlap_specific_fired", "split_overlap_fraction"]
        + [f"{a}_detected" for a in ["verdict_checklist", "verdict_stock_gate",
                                      "verdict_extended_gate", "verdict_temporal_integrity"]]
    ].mean().to_string())

    print("\n" + "=" * 100)
    print("Supplementary periods: flag rate (BLOCKED or NEEDS_REVIEW) across periods not used by the main battery")
    print("=" * 100)
    for appr in ["verdict_checklist", "verdict_stock_gate", "verdict_extended_gate",
                 "verdict_temporal_integrity"]:
        period_df[f"{appr}_flagged"] = period_df[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
    print(period_df.groupby("dataset")[
        [f"{a}_flagged" for a in ["verdict_checklist", "verdict_stock_gate",
                                   "verdict_extended_gate", "verdict_temporal_integrity"]]
    ].mean().to_string())
