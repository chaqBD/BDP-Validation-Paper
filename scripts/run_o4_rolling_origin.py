"""Run O4 rolling-origin reliability analysis for both datasets (Section 7.4, H5)."""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataco_prep as dc
from src import olist_prep as ol
from src.config import O4_DIR, RANDOM_SEED
from src.rolling_origin import run_rolling_origin, auxiliary_operational_model, regress_reliability


def run_dataco():
    df = dc.load_raw()
    X, y, meta, num, cat = dc.build_population(df)
    drift_cols = ["Days for shipment (scheduled)", "Sales", "Order Item Discount Rate"]
    res = run_rolling_origin(X, y, meta, num, cat, drift_cols,
                              promise_col="scheduled_days", fulfillment_col="actual_fulfillment_days",
                              family="gbm", n_origins=10, seed=RANDOM_SEED)
    res.to_csv(O4_DIR / "dataco_rolling_origin.csv", index=False)
    reg = regress_reliability(res)
    with open(O4_DIR / "dataco_h5_regression.json", "w") as f:
        json.dump(reg, f, indent=2, default=float)
    return res, reg


def run_olist():
    orders, customers, items, payments, products = ol.load_raw()
    X, y, meta, num, cat = ol.build_population(orders, customers, items, payments, products)
    drift_cols = ["promised_days", "total_price", "n_items"]
    res = run_rolling_origin(X, y, meta, num, cat, drift_cols,
                              promise_col="promised_days", fulfillment_col="actual_fulfillment_days",
                              family="gbm", n_origins=10, seed=RANDOM_SEED)
    res.to_csv(O4_DIR / "olist_rolling_origin.csv", index=False)
    reg = regress_reliability(res)
    with open(O4_DIR / "olist_h5_regression.json", "w") as f:
        json.dump(reg, f, indent=2, default=float)

    aux = auxiliary_operational_model(X, y, meta, num, cat, promise_col="promised_days",
                                       fulfillment_col="actual_fulfillment_days",
                                       n_origins=10, seed=RANDOM_SEED)
    aux.to_csv(O4_DIR / "olist_auxiliary_operational_model.csv", index=False)
    return res, reg


if __name__ == "__main__":
    t0 = time.time()
    dataco_res, dataco_reg = run_dataco()
    print("DataCo rolling origin:\n", dataco_res[["origin_index", "auc", "mean_psi",
                                                    "change_lateness_prevalence",
                                                    "change_in_next_auc"]].to_string())
    print("DataCo H5 regression:", dataco_reg)

    olist_res, olist_reg = run_olist()
    print("\nOlist rolling origin:\n", olist_res[["origin_index", "auc", "mean_psi",
                                                    "change_lateness_prevalence",
                                                    "change_in_next_auc"]].to_string())
    print("Olist H5 regression:", olist_reg)
    print(f"\nTotal time {time.time()-t0:.1f}s")
