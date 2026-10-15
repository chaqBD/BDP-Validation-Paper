"""O3 fixed-test-set mechanism experiments (Section 7.2).

The test set is held fixed for a designated evaluation period; only the training
set changes. Four conditions isolate one factor at a time:

  A. Past only                                         -> reference (primary setting)
  B. Past + contemporaneous, test orders excluded       -> effect of time coverage
  C. B + sibling items of a controlled share of test
     orders (0-100%, DataCo only: row = line item)      -> effect of order dependence
  D. Past only + earlier orders from the test customers -> legitimate customer history

For a designated period P, orders falling in P are split into a fixed `test`
sample and a `contemporaneous` pool (orders in P not used for testing). The test
set and its size are identical across conditions A/B/C/D; only the training pool
changes. This operationalises "fixed test set, training set a controlled lever"
for data with a row = order-line-item structure (DataCo). For Olist (row =
order), condition C is not run, consistent with Section 5.1's footnote that
row-level and order-level random splits coincide there.
"""
import numpy as np
import pandas as pd

from .metrics import safe_auc, all_metrics
from .models import build_preprocessor, make_estimator, GRIDS
from sklearn.base import clone
from sklearn.pipeline import Pipeline


def _fit_predict(family, params, preprocessor, X_train, y_train, X_test, seed):
    pipe = Pipeline([("prep", clone(preprocessor)), ("clf", make_estimator(family, params, seed))])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


def run_period_experiment(X, y, meta, period_start, period_end, n_train, family="gbm",
                           params=None, seed=0, test_frac_of_period=0.5,
                           overlap_steps=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                           numeric_cols=None, categorical_cols=None,
                           row_order_col="order_id", customer_col="customer_id",
                           run_condition_c=True):
    """Run conditions A, B, D once and C across `overlap_steps` for one evaluation
    period. Returns a list of result dicts.
    """
    rng = np.random.default_rng(seed)
    params = params or GRIDS[family][-1]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    decision_time = meta["decision_time"].values
    order_ids = meta[row_order_col].values
    customer_ids = meta[customer_col].values

    in_period = (decision_time >= np.datetime64(period_start)) & (decision_time < np.datetime64(period_end))
    before_period = decision_time < np.datetime64(period_start)
    if in_period.sum() < 40 or before_period.sum() < n_train:
        return []

    period_orders = np.unique(order_ids[in_period])
    rng.shuffle(period_orders)
    n_test_orders = max(int(len(period_orders) * test_frac_of_period), 10)
    test_orders = set(period_orders[:n_test_orders])
    contemporaneous_orders = set(period_orders[n_test_orders:])

    is_test_order = np.isin(order_ids, list(test_orders))
    test_idx = np.where(in_period & is_test_order)[0]
    if len(test_idx) < 20:
        return []
    y_test, meta_test = y.values[test_idx], meta.iloc[test_idx]

    # A deterministic "kept" row per test order (the smallest row position),
    # with any remaining rows of that order treated as leak-able siblings (DataCo only).
    test_df = pd.DataFrame({"pos": test_idx, "order_id": order_ids[test_idx]})
    kept_pos = test_df.groupby("order_id")["pos"].min().values
    sibling_pos = np.setdiff1d(test_idx, kept_pos)

    before_idx = np.where(before_period)[0]
    contemp_idx = np.where(in_period & np.isin(order_ids, list(contemporaneous_orders)))[0]
    test_customers = set(customer_ids[test_idx])
    before_same_customer_idx = before_idx[np.isin(customer_ids[before_idx], list(test_customers))]
    before_other_customer_idx = np.setdiff1d(before_idx, before_same_customer_idx)

    def sample(pool, k, rs):
        pool = np.asarray(pool)
        if len(pool) <= k:
            return pool
        return rs.choice(pool, size=k, replace=False)

    results = []

    # Condition A: past only.
    train_A = sample(before_idx, n_train, rng)
    p = _fit_predict(family, params, preprocessor, X.iloc[train_A], y.values[train_A],
                      X.iloc[test_idx], seed)
    results.append(dict(condition="A_past_only", overlap_share=np.nan, n_train=len(train_A),
                         **all_metrics(y_test, p)))

    # Condition B: past + contemporaneous (test orders excluded), zero group overlap.
    n_contemp = min(len(contemp_idx), n_train // 3)
    train_B = np.concatenate([sample(before_idx, n_train - n_contemp, rng), contemp_idx[:n_contemp]])
    p = _fit_predict(family, params, preprocessor, X.iloc[train_B], y.values[train_B],
                      X.iloc[test_idx], seed)
    results.append(dict(condition="B_time_coverage", overlap_share=np.nan, n_train=len(train_B),
                         **all_metrics(y_test, p)))

    # Condition C: B + a controlled share of test orders' sibling rows leaked into training.
    if run_condition_c and len(sibling_pos) >= 5:
        sib_orders = np.unique(order_ids[sibling_pos])
        rng.shuffle(sib_orders)
        for share in overlap_steps:
            n_leak_orders = int(round(share * len(sib_orders)))
            leak_orders = set(sib_orders[:n_leak_orders])
            leak_idx = sibling_pos[np.isin(order_ids[sibling_pos], list(leak_orders))]
            base_budget = max(n_train - len(leak_idx), 0)
            train_C = np.concatenate([sample(train_B, base_budget, rng), leak_idx])
            p = _fit_predict(family, params, preprocessor, X.iloc[train_C], y.values[train_C],
                              X.iloc[test_idx], seed)
            results.append(dict(condition="C_order_dependence", overlap_share=share,
                                 n_train=len(train_C), n_leaked_orders=n_leak_orders,
                                 **all_metrics(y_test, p)))

    # Condition D: past only, with the test customers' own earlier orders guaranteed present.
    n_own_history = min(len(before_same_customer_idx), n_train // 2)
    train_D = np.concatenate([
        sample(before_same_customer_idx, n_own_history, rng),
        sample(before_other_customer_idx, n_train - n_own_history, rng),
    ])
    p = _fit_predict(family, params, preprocessor, X.iloc[train_D], y.values[train_D],
                      X.iloc[test_idx], seed)
    results.append(dict(condition="D_customer_history", overlap_share=np.nan, n_train=len(train_D),
                         n_own_history_rows=n_own_history, **all_metrics(y_test, p)))

    for r in results:
        r["period_start"] = str(period_start)
        r["period_end"] = str(period_end)
        r["n_test"] = len(test_idx)
        r["n_test_orders"] = len(test_orders)
    return results
