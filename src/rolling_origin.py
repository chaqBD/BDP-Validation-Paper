"""O4: reliability over time and changes in the service promise (Section 7.4, H5).

At each rolling-origin forecast origin, records feature drift (PSI, standardised
mean shift), the change in lateness prevalence, the change in the promised
window and in fulfilment time, and the following period's change in AUC and
calibration. Associations are summarised with an origin-level regression.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .metrics import all_metrics
from .models import build_preprocessor, make_estimator, GRIDS
from .splitters import rolling_origin_splits
from .stats_tests import population_stability_index, standardized_mean_shift, slope_test


def run_rolling_origin(X, y, meta, numeric_cols, categorical_cols, drift_cols,
                        promise_col, fulfillment_col, family="gbm", n_origins=10,
                        min_train_frac=0.4, test_frac_per_origin=0.06, seed=0):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    params = GRIDS[family][-1]
    splits = rolling_origin_splits(meta, n_origins=n_origins, min_train_frac=min_train_frac,
                                    test_frac_per_origin=test_frac_per_origin)

    rows = []
    for s in splits:
        train_idx, test_idx = s["train_idx"], s["test_idx"]
        pipe = Pipeline([("prep", clone(preprocessor)),
                          ("clf", make_estimator(family, params, seed=seed))])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        p_test = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        m = all_metrics(y.iloc[test_idx].values, p_test)

        train_meta, test_meta = meta.iloc[train_idx], meta.iloc[test_idx]
        train_X, test_X = X.iloc[train_idx], X.iloc[test_idx]

        psi_vals, smd_vals = {}, {}
        for col in drift_cols:
            psi_vals[col] = population_stability_index(train_X[col].values, test_X[col].values)
            smd_vals[col] = standardized_mean_shift(train_X[col].values, test_X[col].values)

        row = {
            "origin_index": s["origin_index"], "origin_time": str(s["origin_time"]),
            "horizon_end": str(s["horizon_end"]), "n_train": len(train_idx), "n_test": len(test_idx),
            **m,
            "mean_psi": float(np.nanmean(list(psi_vals.values()))),
            "mean_abs_smd": float(np.nanmean(np.abs(list(smd_vals.values())))),
            "lateness_prevalence_train": float(y.iloc[train_idx].mean()),
            "lateness_prevalence_test": float(y.iloc[test_idx].mean()),
            "promised_mean_train": float(train_meta[promise_col].mean()),
            "promised_mean_test": float(test_meta[promise_col].mean()),
            "fulfillment_mean_train": float(train_meta[fulfillment_col].mean()),
            "fulfillment_mean_test": float(test_meta[fulfillment_col].mean()),
        }
        row["delay_vs_promise_test"] = row["fulfillment_mean_test"] - row["promised_mean_test"]
        row["delay_vs_promise_train"] = row["fulfillment_mean_train"] - row["promised_mean_train"]
        row["change_lateness_prevalence"] = (row["lateness_prevalence_test"]
                                              - row["lateness_prevalence_train"])
        row["change_promised_window"] = row["promised_mean_test"] - row["promised_mean_train"]
        row["change_fulfillment_time"] = row["fulfillment_mean_test"] - row["fulfillment_mean_train"]
        for col in drift_cols:
            row[f"psi_{col}"] = psi_vals[col]
            row[f"smd_{col}"] = smd_vals[col]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("origin_index").reset_index(drop=True)
    df["next_auc"] = df["auc"].shift(-1)
    df["next_cal_in_large"] = df["cal_in_large"].shift(-1)
    df["change_in_next_auc"] = df["next_auc"] - df["auc"]
    df["change_in_next_cal_in_large"] = df["next_cal_in_large"].abs() - df["cal_in_large"].abs()
    return df


def auxiliary_operational_model(X, y, meta, numeric_cols, categorical_cols, promise_col,
                                 fulfillment_col, exclude_cols=(), n_origins=10,
                                 min_train_frac=0.4, test_frac_per_origin=0.06, seed=0):
    """Olist-only (Section 7.4): a fulfilment-duration regressor trained on
    operational features (excluding the promise) separates genuine operational
    drift (the regressor's error grows over time) from a pure change in the
    promised window (the regressor stays accurate while the promise itself moves).
    """
    op_numeric = [c for c in numeric_cols if c not in exclude_cols and c != promise_col]
    preprocessor = build_preprocessor(op_numeric, categorical_cols)
    splits = rolling_origin_splits(meta, n_origins=n_origins, min_train_frac=min_train_frac,
                                    test_frac_per_origin=test_frac_per_origin)
    rows = []
    for s in splits:
        train_idx, test_idx = s["train_idx"], s["test_idx"]
        pipe = Pipeline([("prep", clone(preprocessor)),
                          ("reg", RandomForestRegressor(n_estimators=200, max_depth=10,
                                                         min_samples_leaf=5, n_jobs=-1,
                                                         random_state=seed))])
        fulfil_train = meta[fulfillment_col].values[train_idx]
        fulfil_test = meta[fulfillment_col].values[test_idx]
        pipe.fit(X.iloc[train_idx][op_numeric + categorical_cols], fulfil_train)
        pred = pipe.predict(X.iloc[test_idx][op_numeric + categorical_cols])
        mae = float(np.mean(np.abs(pred - fulfil_test)))
        rows.append({
            "origin_index": s["origin_index"], "origin_time": str(s["origin_time"]),
            "operational_model_mae": mae,
            "promised_mean_test": float(meta[promise_col].values[test_idx].mean()),
            "fulfillment_mean_test": float(fulfil_test.mean()),
        })
    return pd.DataFrame(rows).sort_values("origin_index").reset_index(drop=True)


def regress_reliability(df):
    """Origin-level regressions for H5: feature drift (PSI) vs change in next-period
    AUC; change in lateness prevalence vs change in next-period calibration-in-the-large.
    """
    clean = df.dropna(subset=["change_in_next_auc", "mean_psi", "change_lateness_prevalence"])
    out = {}
    if len(clean) >= 4:
        out["psi_vs_next_auc_change"] = slope_test(clean["mean_psi"], clean["change_in_next_auc"])
        out["lateness_change_vs_next_cal_change"] = slope_test(
            clean["change_lateness_prevalence"], clean["change_in_next_cal_in_large"])
    return out
