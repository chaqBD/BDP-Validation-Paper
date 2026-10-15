"""Run O3 fixed-test-set mechanism experiments (Section 7.2) for both datasets."""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import dataco_prep as dc
from src import olist_prep as ol
from src.config import O3_MECH_DIR, RANDOM_SEED
from src.mechanism_experiments import run_period_experiment

DATACO_PERIODS = [
    ("2017-03-01", "2017-04-01"),
    ("2017-05-01", "2017-06-01"),
    ("2017-07-01", "2017-08-01"),
    ("2017-09-01", "2017-09-30"),
]
OLIST_PERIODS = [
    ("2017-11-01", "2017-12-01"),
    ("2018-02-01", "2018-03-01"),
    ("2018-05-01", "2018-06-01"),
]


def run_dataco():
    df = dc.load_raw()
    X, y, meta, num, cat = dc.build_population(df)
    all_rows = []
    for start, end in DATACO_PERIODS:
        t0 = time.time()
        res = run_period_experiment(
            X, y, meta, start, end, n_train=6000, family="gbm", seed=RANDOM_SEED,
            overlap_steps=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            numeric_cols=num, categorical_cols=cat, run_condition_c=True)
        print(f"DataCo period {start}..{end}: {len(res)} rows in {time.time()-t0:.1f}s")
        all_rows.extend(res)
    out = pd.DataFrame(all_rows)
    out["dataset"] = "dataco"
    out.to_csv(O3_MECH_DIR / "dataco_mechanism_results.csv", index=False)
    return out


def run_olist():
    orders, customers, items, payments, products = ol.load_raw()
    X, y, meta, num, cat = ol.build_population(orders, customers, items, payments, products)
    all_rows = []
    for start, end in OLIST_PERIODS:
        t0 = time.time()
        res = run_period_experiment(
            X, y, meta, start, end, n_train=4000, family="gbm", seed=RANDOM_SEED,
            numeric_cols=num, categorical_cols=cat, run_condition_c=False)
        print(f"Olist period {start}..{end}: {len(res)} rows in {time.time()-t0:.1f}s")
        all_rows.extend(res)
    out = pd.DataFrame(all_rows)
    out["dataset"] = "olist"
    out.to_csv(O3_MECH_DIR / "olist_mechanism_results.csv", index=False)
    return out


if __name__ == "__main__":
    t0 = time.time()
    dataco_res = run_dataco()
    olist_res = run_olist()
    print(f"Total time {time.time()-t0:.1f}s")
    print(dataco_res.groupby("condition")["auc"].mean())
    print(olist_res.groupby("condition")["auc"].mean())
