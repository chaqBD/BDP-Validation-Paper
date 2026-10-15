"""Run O2 (validation-design comparison) on Olist. Saves results CSV + predictions."""
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import olist_prep as ol
from src.config import O2_DIR, RANDOM_SEED
from src.models import BaselineModel
from src.o2_pipeline import run_o2

if __name__ == "__main__":
    t0 = time.time()
    print("Loading Olist...")
    orders, customers, items, payments, products = ol.load_raw()
    X, y, meta, num, cat = ol.build_population(orders, customers, items, payments, products)
    print(f"Population: {X.shape}, late rate {y.mean():.3f}, n_customers={meta.customer_id.nunique()}")

    baselines = {
        "baseline_promised_days": lambda: BaselineModel(["promised_days"], categorical=False),
        "baseline_customer_region": lambda: BaselineModel(["customer_state"], categorical=True),
    }

    results, preds, design_meta, gate_summary = run_o2(
        "olist", X, y, meta, num, cat, baselines,
        test_size=0.2, n_boot=400, seed=RANDOM_SEED)

    results.to_csv(O2_DIR / "olist_results.csv", index=False)
    gate_summary.to_csv(O2_DIR / "olist_gate_summary.csv", index=False)
    with open(O2_DIR / "olist_preds.pkl", "wb") as f:
        pickle.dump(preds, f)
    import json
    with open(O2_DIR / "olist_design_meta.json", "w") as f:
        json.dump(design_meta, f, indent=2)

    print(f"Done in {time.time()-t0:.1f}s. Saved to {O2_DIR}")
    print(results[["design", "model", "auc", "auc_ci_low", "auc_ci_high", "brier", "cal_slope"]]
          .to_string())
    print("\nGate summary:\n", gate_summary.to_string())
