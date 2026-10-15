"""Held-out evaluation of the group-overlap extension and the temporal-integrity
comparator, beyond the single constructed instance per defect type used in the main
battery (`run_release_review.py`). Two independent extensions of that battery:

1. Severity/seed sweep on the exact-duplicate defect: the only defect whose
   severity is a free parameter (the fraction of test rows also duplicated into
   training) and whose instances can be redrawn under independent seeds. Five
   severities x three seeds each, never seen by any threshold tuned in this study
   (`src.release_review`'s frozen thresholds predate this sweep).
2. Every rolling-origin period NOT already used by the two hand-picked scenario
   cases in the main battery (8 of the 10 origins per dataset), to check the
   false-positive rate of the extension and the temporal-integrity check across
   genuinely held-out legitimate periods, not just the two periods the main
   battery happened to select.

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

OUT_DIR = RESULTS_DIR / "release_review"
FAMILY = "gbm"

SEVERITIES = [0.02, 0.05, 0.1, 0.2, 0.3]
SEEDS = [101, 202, 303]

# The two origin indices already used as the main battery's hand-picked scenario
# cases, per dataset -- excluded here so every row below is a genuinely new,
# previously unevaluated period.
MAIN_BATTERY_ORIGINS = {"dataco": {9, 1}, "olist": {9, 3}}
N_ORIGINS = 10


if __name__ == "__main__":
    severity_rows = []
    period_rows = []

    for dataset in ["dataco", "olist"]:
        ref = _reference(dataset)

        # 1. Severity/seed sweep on exact-duplicate.
        t0 = time.time()
        for frac in SEVERITIES:
            for seed in SEEDS:
                case = case_exact_duplicate(ref, frac=frac, seed=seed)
                row = run_one_case(dataset, "defect_exact_duplicate", "exact_duplicate",
                                    True, case, ref, FAMILY)
                row["severity_frac"] = frac
                row["seed"] = seed
                severity_rows.append(row)
        print(f"[{dataset}] severity/seed sweep done in {time.time()-t0:.1f}s "
              f"({len(SEVERITIES)*len(SEEDS)} instances)")

        # 2. Every held-out rolling-origin period not used by the main battery.
        t0 = time.time()
        held_out_origins = [i for i in range(N_ORIGINS) if i not in MAIN_BATTERY_ORIGINS[dataset]]
        for origin_index in held_out_origins:
            case = case_scenario(dataset, origin_index=origin_index)
            row = run_one_case(dataset, f"heldout_period_{origin_index}", "scenario_heldout",
                                False, case, ref, FAMILY)
            row["origin_index"] = origin_index
            period_rows.append(row)
        print(f"[{dataset}] held-out period sweep done in {time.time()-t0:.1f}s "
              f"({len(held_out_origins)} periods)")

    severity_df = pd.DataFrame(severity_rows)
    period_df = pd.DataFrame(period_rows)
    severity_df.to_csv(OUT_DIR / "heldout_severity_sweep.csv", index=False)
    period_df.to_csv(OUT_DIR / "heldout_period_sweep.csv", index=False)
    print("\nSaved to", OUT_DIR / "heldout_severity_sweep.csv")
    print("Saved to", OUT_DIR / "heldout_period_sweep.csv")

    print("\n" + "=" * 100)
    print("Severity sweep: detection rate (BLOCKED or NEEDS_REVIEW) by severity, across seeds")
    print("=" * 100)
    for appr in ["verdict_checklist", "verdict_stock_gate", "verdict_extended_gate",
                 "verdict_temporal_integrity"]:
        severity_df[f"{appr}_detected"] = severity_df[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
    print(severity_df.groupby(["dataset", "severity_frac"])[
        [f"{a}_detected" for a in ["verdict_checklist", "verdict_stock_gate",
                                    "verdict_extended_gate", "verdict_temporal_integrity"]]
    ].mean().to_string())

    print("\n" + "=" * 100)
    print("Held-out periods: false-positive rate (BLOCKED or NEEDS_REVIEW) across genuinely new periods")
    print("=" * 100)
    for appr in ["verdict_checklist", "verdict_stock_gate", "verdict_extended_gate",
                 "verdict_temporal_integrity"]:
        period_df[f"{appr}_flagged"] = period_df[appr].isin(["BLOCKED", "NEEDS_REVIEW"])
    print(period_df.groupby("dataset")[
        [f"{a}_flagged" for a in ["verdict_checklist", "verdict_stock_gate",
                                   "verdict_extended_gate", "verdict_temporal_integrity"]]
    ].mean().to_string())
