"""RQ1 (H1, H2) and RQ4 (H6) hypothesis tests from the O2 design-comparison runs
(Section 4). Loads the predictions saved by run_o2_dataco.py / run_o2_olist.py.
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bootstrap import (paired_cluster_bootstrap_difference, equivalence_test, holm_adjust,
                            bootstrap_pvalue_two_sided)
from src.config import O2_DIR, DELTA_AUC
from src.metrics import safe_auc
from src.stats_tests import kendall_tau_with_bootstrap

MODEL_FAMILIES = ["logreg", "rf", "gbm"]


def load(dataset):
    with open(O2_DIR / f"{dataset}_preds.pkl", "rb") as f:
        preds = pickle.load(f)
    results = pd.read_csv(O2_DIR / f"{dataset}_results.csv")
    return preds, results


def h1_dataco(preds, n_boot=1000, seed=0):
    """H1: row-level random AUC exceeds order-grouped AUC by more than delta, DataCo."""
    rows = []
    for family in MODEL_FAMILIES:
        y_a, p_a, c_a = preds[("row_random", family)]
        y_b, p_b, c_b = preds[("order_grouped", family)]
        diff, lo, hi, diffs = paired_cluster_bootstrap_difference(
            y_a, p_a, c_a, safe_auc, y_b, p_b, c_b, n_boot=n_boot, seed=seed)
        verdict = equivalence_test(diff, lo, hi, DELTA_AUC)
        pval = bootstrap_pvalue_two_sided(diffs)
        rows.append({"hypothesis": "H1", "dataset": "dataco", "model": family,
                      "contrast": "row_random_minus_order_grouped_auc",
                      "diff": diff, "ci_low": lo, "ci_high": hi, "delta": DELTA_AUC,
                      "verdict": verdict, "pvalue_diff_ne_zero": pval})
    return rows


def h2_olist(preds, n_boot=1000, seed=0):
    """H2: chronological AUC is lower than order-level random AUC by more than delta, Olist."""
    rows = []
    for family in MODEL_FAMILIES:
        y_a, p_a, c_a = preds[("chronological_primary", family)]
        y_b, p_b, c_b = preds[("order_grouped", family)]
        diff, lo, hi, diffs = paired_cluster_bootstrap_difference(
            y_a, p_a, c_a, safe_auc, y_b, p_b, c_b, n_boot=n_boot, seed=seed)
        verdict = equivalence_test(diff, lo, hi, DELTA_AUC)
        pval = bootstrap_pvalue_two_sided(diffs)
        rows.append({"hypothesis": "H2", "dataset": "olist", "model": family,
                      "contrast": "chronological_minus_order_grouped_auc",
                      "diff": diff, "ci_low": lo, "ci_high": hi, "delta": DELTA_AUC,
                      "verdict": verdict, "pvalue_diff_ne_zero": pval})
    return rows


def h6_ranking(preds, dataset, primary_design, n_boot=1000, seed=0):
    """H6: model rankings under row-level random validation differ from rankings under
    the primary design."""
    row_aucs, primary_aucs = [], []
    for family in MODEL_FAMILIES:
        y, p, c = preds[("row_random", family)]
        row_aucs.append(safe_auc(y, p))
        y2, p2, c2 = preds[(primary_design, family)]
        primary_aucs.append(safe_auc(y2, p2))
    row_ranks = pd.Series(row_aucs).rank(ascending=False).values
    primary_ranks = pd.Series(primary_aucs).rank(ascending=False).values
    res = kendall_tau_with_bootstrap(row_ranks, primary_ranks, n_boot=n_boot, seed=seed)
    res.update({"hypothesis": "H6", "dataset": dataset, "models": MODEL_FAMILIES,
                "row_random_aucs": row_aucs, "row_random_ranks": row_ranks.tolist(),
                "primary_design": primary_design, "primary_aucs": primary_aucs,
                "primary_ranks": primary_ranks.tolist()})
    return res


if __name__ == "__main__":
    dataco_preds, dataco_results = load("dataco")
    olist_preds, olist_results = load("olist")

    all_rows = []
    all_rows += h1_dataco(dataco_preds)
    all_rows += h2_olist(olist_preds)

    # Holm's step-down correction applied to genuine bootstrap p-values for the two-sided
    # null H0: diff = 0 (NOT the same null as the registered equivalence test, which is a
    # CI-vs-margin decision rule and does not itself require a p-value; this column is a
    # supplementary multiplicity-controlled significance statement, reported alongside,
    # not in place of, the primary equivalence verdict).
    pvals = [r["pvalue_diff_ne_zero"] for r in all_rows]
    adj = holm_adjust(pvals)
    for r, a in zip(all_rows, adj):
        r["holm_adjusted_p_diff_ne_zero"] = a

    h_df = pd.DataFrame(all_rows)
    h_df.to_csv(O2_DIR / "hypothesis_tests_h1_h2.csv", index=False)
    print(h_df.to_string())

    h6_dataco = h6_ranking(dataco_preds, "dataco", "chronological_primary")
    h6_olist = h6_ranking(olist_preds, "olist", "chronological_primary")
    with open(O2_DIR / "hypothesis_test_h6.json", "w") as f:
        json.dump({"dataco": h6_dataco, "olist": h6_olist}, f, indent=2, default=float)
    print("\nH6 (Kendall's tau, row-random vs chronological-primary rankings):")
    print("DataCo:", h6_dataco)
    print("Olist:", h6_olist)
