"""O1: fix the prediction task and audit the data for both datasets (Section 5, 6)."""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataco_prep as dc
from src import olist_prep as ol
from src.config import AUDIT_DIR

if __name__ == "__main__":
    df = dc.load_raw()
    dataco_audit = dc.audit(df)
    with open(AUDIT_DIR / "dataco_audit.json", "w") as f:
        json.dump(dataco_audit, f, indent=2)
    print("DataCo audit:", json.dumps(dataco_audit, indent=2))

    X, y, meta, num, cat = dc.build_population(df)
    dataco_pop_summary = {
        "n_rows": int(len(X)), "n_orders": int(meta.order_id.nunique()),
        "n_customers": int(meta.customer_id.nunique()), "late_rate": float(y.mean()),
        "date_min": str(meta.decision_time.min()), "date_max": str(meta.decision_time.max()),
        "n_features": int(X.shape[1]), "numeric_features": num, "categorical_features": cat,
    }
    with open(AUDIT_DIR / "dataco_population_summary.json", "w") as f:
        json.dump(dataco_pop_summary, f, indent=2)
    print("DataCo population:", json.dumps(dataco_pop_summary, indent=2))

    orders, customers, items, payments, products = ol.load_raw()
    olist_audit, monthly = ol.audit(orders, items)
    with open(AUDIT_DIR / "olist_audit.json", "w") as f:
        json.dump(olist_audit, f, indent=2)
    monthly.to_csv(AUDIT_DIR / "olist_monthly_late_rate.csv")
    print("\nOlist audit:", json.dumps(olist_audit, indent=2))

    X, y, meta, num, cat = ol.build_population(orders, customers, items, payments, products)
    olist_pop_summary = {
        "n_rows": int(len(X)), "n_orders": int(meta.order_id.nunique()),
        "n_customers": int(meta.customer_id.nunique()), "late_rate": float(y.mean()),
        "date_min": str(meta.decision_time.min()), "date_max": str(meta.decision_time.max()),
        "n_features": int(X.shape[1]), "numeric_features": num, "categorical_features": cat,
    }
    with open(AUDIT_DIR / "olist_population_summary.json", "w") as f:
        json.dump(olist_pop_summary, f, indent=2)
    print("\nOlist population:", json.dumps(olist_pop_summary, indent=2))
