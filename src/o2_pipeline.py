"""O2: sensitivity of performance estimates to the validation design (Section 7.1).

Compares random row-level, order-grouped, customer-grouped, chronological (primary
and secondary new-customer estimand) designs across three model families and two
per-dataset baselines, with matched training size and cluster-bootstrap intervals.
"""
import time

import numpy as np
import pandas as pd

from .bootstrap import cluster_bootstrap_metric
from .config import RANDOM_SEED, DELTA_AUC, O2_DIR
from .gate_integration import run_gate_report
from .metrics import all_metrics, safe_auc
from .models import build_preprocessor, time_aware_tune
from .splitters import (split_row_random, split_group_random, split_chronological,
                         split_chronological_new_customers, match_training_size)

MODEL_FAMILIES = ["logreg", "rf", "gbm"]


def build_designs(meta, test_size=0.2, seed=RANDOM_SEED):
    designs = {}
    designs["row_random"] = split_row_random(meta, test_size, seed)
    designs["order_grouped"] = split_group_random(meta, "order_id", test_size, seed)
    designs["customer_grouped"] = split_group_random(meta, "customer_id", test_size, seed)
    train_chrono, test_chrono, cutoff = split_chronological(meta, test_size)
    designs["chronological_primary"] = (train_chrono, test_chrono)

    train_chrono2, test_new, test_returning, cutoff2 = split_chronological_new_customers(
        meta, test_size=test_size)
    designs["chronological_new_customers"] = (train_chrono2, test_new)

    match_keys = ["row_random", "order_grouped", "customer_grouped", "chronological_primary"]
    matched, min_orders = match_training_size(
        {k: designs[k][0] for k in match_keys}, meta, group_col="order_id", seed=seed)
    for k in match_keys:
        designs[k] = (matched[k], designs[k][1])
    designs["chronological_new_customers"] = (matched["chronological_primary"],
                                                designs["chronological_new_customers"][1])
    return designs, {"min_orders": int(min_orders), "cutoff": str(cutoff)}


def run_o2(dataset_name, X, y, meta, numeric_cols, categorical_cols, baseline_factories,
           test_size=0.2, n_boot=300, seed=RANDOM_SEED, verbose=True, run_gate=True):
    """baseline_factories: dict[name -> callable() -> fitted-compatible unfit estimator
    object implementing .fit(X,y) and .predict_proba(X)->(n,) via models.BaselineModel].

    run_gate: when True (Section 7.7), every fitted (design, model) pipeline is also
    passed through the BDP Model Gate, which produces a standardised PASS /
    NEEDS_REVIEW / BLOCKED validation/calibration/drift report, saved to
    results/o2_design_comparison/gate_reports/<dataset>/<design>_<model>.json.
    Reported descriptively alongside our own metrics; it does not replace them.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    designs, design_meta = build_designs(meta, test_size=test_size, seed=seed)
    gate_dir = O2_DIR / "gate_reports" / dataset_name
    strategy_for_design = {
        "row_random": "random_split",
        "order_grouped": "grouped_split",
        "customer_grouped": "grouped_split",
        "chronological_primary": "out_of_time",
        "chronological_new_customers": "out_of_time",
    }

    rows = []
    preds_store = {}
    gate_rows = []

    for design_name, (train_idx, test_idx) in designs.items():
        if len(test_idx) < 20 or len(train_idx) < 50:
            continue
        t0 = time.time()
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        order_time_train = pd.Series(meta["decision_time"].values[train_idx])
        cluster_ids_test = meta["order_id"].values[test_idx]

        n_train_orders = len(np.unique(meta["order_id"].values[train_idx]))

        for family in MODEL_FAMILIES:
            pipe, best_params, inner_score = time_aware_tune(
                family, X_train, y_train, order_time_train, preprocessor,
                n_splits=2, seed=seed)
            p_test = pipe.predict_proba(X_test)[:, 1]
            m = all_metrics(y_test.values, p_test)
            auc_pt, auc_lo, auc_hi, _ = cluster_bootstrap_metric(
                y_test.values, p_test, cluster_ids_test, safe_auc, n_boot=n_boot, seed=seed)
            rows.append({
                "dataset": dataset_name, "design": design_name, "model": family,
                "is_baseline": False, "n_train_rows": len(train_idx),
                "n_train_orders": n_train_orders, "n_test_rows": len(test_idx),
                "best_params": str(best_params), "inner_cv_auc": inner_score,
                **m, "auc_ci_low": auc_lo, "auc_ci_high": auc_hi,
            })
            preds_store[(design_name, family)] = (y_test.values, p_test, cluster_ids_test)

            if run_gate:
                gres = run_gate_report(pipe, X_train, X_test, y_test.values, p_test,
                                        gate_dir, f"{design_name}_{family}",
                                        dataset_name=dataset_name,
                                        validation_strategy=strategy_for_design[design_name])
                gate_rows.append({"dataset": dataset_name, "design": design_name,
                                   "model": family, "gate_status": gres.get("gate_status")})

        for bname, factory in baseline_factories.items():
            baseline = factory()
            baseline.fit(X_train, y_train)
            p_test = baseline.predict_proba(X_test)
            m = all_metrics(y_test.values, p_test)
            auc_pt, auc_lo, auc_hi, _ = cluster_bootstrap_metric(
                y_test.values, p_test, cluster_ids_test, safe_auc, n_boot=n_boot, seed=seed)
            rows.append({
                "dataset": dataset_name, "design": design_name, "model": bname,
                "is_baseline": True, "n_train_rows": len(train_idx),
                "n_train_orders": n_train_orders, "n_test_rows": len(test_idx),
                "best_params": None, "inner_cv_auc": np.nan,
                **m, "auc_ci_low": auc_lo, "auc_ci_high": auc_hi,
            })
            preds_store[(design_name, bname)] = (y_test.values, p_test, cluster_ids_test)

            if run_gate:
                gres = run_gate_report(baseline.pipe, X_train[baseline.feature_cols],
                                        X_test[baseline.feature_cols], y_test.values, p_test,
                                        gate_dir, f"{design_name}_{bname}",
                                        dataset_name=dataset_name,
                                        validation_strategy=strategy_for_design[design_name])
                gate_rows.append({"dataset": dataset_name, "design": design_name,
                                   "model": bname, "gate_status": gres.get("gate_status")})

        if verbose:
            print(f"[{dataset_name}] design={design_name} done in {time.time()-t0:.1f}s "
                  f"(train_rows={len(train_idx)}, train_orders={n_train_orders})")

    results = pd.DataFrame(rows)
    gate_summary = pd.DataFrame(gate_rows)
    return results, preds_store, design_meta, gate_summary
