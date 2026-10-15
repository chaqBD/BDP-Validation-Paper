"""O3 simulation with known mechanisms (Section 7.3, H4).

Semi-synthetic outcomes are generated from a known data-generating process with
four factors, applied singly and in a full factorial combination:
  W - within-order correlation (order-level random effect shared by siblings)
  S - covariate shift at test time (mean shift of an operational feature)
  R - base-rate shift at test time (intercept shift -> prevalence change)
  P - promise change at test time (coefficient drift on the "promise" feature)

Each simulated population has orders assigned sequential decision times; the last
`future_frac` of orders are the "future" regime where S/R/P (if switched on) are
applied. Three validation-design test sets are built from the same population:
  - row-random: random rows, ignoring order structure and time
  - order-grouped random: random orders, ignoring time
  - chronological: past orders train, future orders test

The random-vs-grouped contrast should track W; the grouped-vs-chronological
contrast should track S/R/P (H4). Recovery error is the gap between a contrast
estimated from a modest-N replicate and the same contrast computed from a very
large reference population (treated as ~the population truth).
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .metrics import safe_auc


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate_population(n_orders, w_sigma, covariate_shift_mag, base_rate_shift_logit,
                         promise_coef_shift, future_frac=0.3, beta0=-0.2, b1=0.5, b2=0.7,
                         seed=0):
    rng = np.random.default_rng(seed)
    items_per_order = rng.poisson(1.75, size=n_orders) + 1  # mean ~2.75, like DataCo
    order_id = np.repeat(np.arange(n_orders), items_per_order)
    n_rows = len(order_id)

    order_time = np.arange(n_orders)
    row_time = order_time[order_id]
    cutoff = int(n_orders * (1 - future_frac))
    is_future = row_time >= cutoff

    X1 = rng.normal(0, 1, n_rows)
    # X2 is an order-level covariate shared by every sibling row (like DataCo's
    # shipping mode / scheduled days, which are set once per order, not per item).
    X2 = rng.normal(0, 1, n_orders)[order_id]
    X1 = np.where(is_future, X1 + covariate_shift_mag, X1)

    order_effect = rng.normal(0, w_sigma, n_orders)[order_id] if w_sigma > 0 else np.zeros(n_rows)
    # "oid" is the literal order identifier. When w_sigma = 0 the outcome does not
    # depend on order_effect at all, so oid carries no information (no leakage
    # channel exists). When w_sigma > 0, a flexible model that has seen a sibling
    # row's (oid, y) pair in training can look up oid for a test sibling and infer
    # its order_effect almost exactly -- a lookup that only works for orders the
    # model has already seen, and so does NOT generalise to unseen orders. This
    # mirrors the structural leakage measured in the real DataCo mechanism
    # experiment (shared order-level covariates + an order-level label).
    oid = order_id.astype(float)

    beta0_eff = np.where(is_future, beta0 + base_rate_shift_logit, beta0)
    b2_eff = np.where(is_future, b2 + promise_coef_shift, b2)

    eta = beta0_eff + b1 * X1 + b2_eff * X2 + order_effect
    p = _sigmoid(eta)
    y = rng.binomial(1, p)

    X = pd.DataFrame({"x1": X1, "x2": X2, "oid": oid})
    meta = pd.DataFrame({"order_id": order_id, "row_time": row_time, "is_future": is_future})
    return X, pd.Series(y), meta


def _fit_auc(X_train, y_train, X_test, y_test):
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return np.nan
    # A flexible model, matching the gradient-boosted model used in the real
    # DataCo mechanism experiment: enough capacity to exploit near-duplicate
    # sibling rows when they straddle train/test, which a smooth linear model
    # (correctly specified on the *population* relationship) would not.
    clf = RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=1,
                                  n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    return safe_auc(y_test.values, p)


def compute_contrasts(X, y, meta, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)

    # row-random
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    row_test, row_train = idx[:n_test], idx[n_test:]
    auc_row = _fit_auc(X.iloc[row_train], y.iloc[row_train], X.iloc[row_test], y.iloc[row_test])

    # order-grouped random
    orders = meta["order_id"].unique()
    rng.shuffle(orders)
    n_test_orders = max(int(len(orders) * test_frac), 1)
    test_orders = set(orders[:n_test_orders])
    is_test = meta["order_id"].isin(test_orders).values
    auc_grouped = _fit_auc(X.iloc[~is_test], y.iloc[~is_test], X.iloc[is_test], y.iloc[is_test])

    # chronological
    is_future = meta["is_future"].values
    auc_chrono = _fit_auc(X.iloc[~is_future], y.iloc[~is_future], X.iloc[is_future], y.iloc[is_future])

    return {
        "auc_row_random": auc_row, "auc_order_grouped": auc_grouped, "auc_chronological": auc_chrono,
        "contrast_random_vs_grouped": auc_row - auc_grouped,
        "contrast_grouped_vs_chronological": auc_grouped - auc_chrono,
    }


def run_factorial(n_orders_rep=3200, n_orders_truth=40000, n_reps=5, seed=0,
                   w_sigma_on=3.0, covariate_shift_on=0.8, base_rate_shift_on=0.6,
                   promise_coef_shift_on=-0.6):
    factors = ["W", "S", "R", "P"]
    rows = []
    for combo in itertools.product([0, 1], repeat=4):
        cell = dict(zip(factors, combo))
        w_sigma = w_sigma_on if cell["W"] else 0.0
        cov_shift = covariate_shift_on if cell["S"] else 0.0
        base_shift = base_rate_shift_on if cell["R"] else 0.0
        promise_shift = promise_coef_shift_on if cell["P"] else 0.0

        X_t, y_t, meta_t = simulate_population(
            n_orders_truth, w_sigma, cov_shift, base_shift, promise_shift, seed=seed + 9000)
        truth = compute_contrasts(X_t, y_t, meta_t, seed=seed + 9001)

        for rep in range(n_reps):
            rep_seed = seed + 100 * sum(c * (10 ** i) for i, c in enumerate(combo)) + rep
            X_r, y_r, meta_r = simulate_population(
                n_orders_rep, w_sigma, cov_shift, base_shift, promise_shift, seed=rep_seed)
            est = compute_contrasts(X_r, y_r, meta_r, seed=rep_seed + 1)
            rows.append({
                **cell, "rep": rep,
                **{f"est_{k}": v for k, v in est.items()},
                **{f"truth_{k}": v for k, v in truth.items()},
                "recovery_error_random_vs_grouped":
                    est["contrast_random_vs_grouped"] - truth["contrast_random_vs_grouped"],
                "recovery_error_grouped_vs_chronological":
                    est["contrast_grouped_vs_chronological"] - truth["contrast_grouped_vs_chronological"],
            })
    return pd.DataFrame(rows)
