"""Model families, preprocessing and time-aware inner-validation tuning (Section 7.1).

Three model families (regularised logistic regression, random forest, gradient
boosting) plus per-dataset baselines. Preprocessing, encoding and calibration are
fit within each training set only (no leakage across the train/test boundary).
"""
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier

from .config import RANDOM_SEED
from .metrics import safe_auc


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


# Small, fixed hyperparameter grids. Kept deliberately narrow so the same tuning
# budget (number of candidate configurations) is spent for every model family,
# every validation design and every dataset (Section 7.1, "equal footing").
GRIDS = {
    "logreg": [
        {"C": 0.1},
        {"C": 3.0},
    ],
    "rf": [
        {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 20},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
    ],
    "gbm": [
        {"n_estimators": 150, "num_leaves": 15, "learning_rate": 0.1},
        {"n_estimators": 250, "num_leaves": 31, "learning_rate": 0.05},
    ],
}


def make_estimator(family, params, seed=RANDOM_SEED):
    if family == "logreg":
        return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000,
                                   C=params["C"], random_state=seed)
    if family == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if family == "gbm":
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1,
                               objective="binary", **params)
    raise ValueError(family)


def time_aware_tune(family, X, y, order_time, preprocessor, n_splits=3, seed=RANDOM_SEED):
    """Pick the grid config with the best mean inner-validation AUC under a
    TimeSeriesSplit on rows sorted by decision time (time-aware inner validation,
    Section 7.1). Returns the fitted Pipeline on the *full* training set with the
    winning config.
    """
    order = np.argsort(order_time.values if hasattr(order_time, "values") else order_time)
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score, best_params = -np.inf, GRIDS[family][0]
    for params in GRIDS[family]:
        scores = []
        for tr_idx, va_idx in tscv.split(X_sorted):
            pipe = Pipeline([
                ("prep", clone(preprocessor)),
                ("clf", make_estimator(family, params, seed=seed)),
            ])
            pipe.fit(X_sorted.iloc[tr_idx], y_sorted[tr_idx])
            p = pipe.predict_proba(X_sorted.iloc[va_idx])[:, 1]
            scores.append(safe_auc(y_sorted[va_idx], p))
        scores = [s for s in scores if not np.isnan(s)]
        mean_score = np.mean(scores) if scores else -np.inf
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    final_pipe = Pipeline([
        ("prep", clone(preprocessor)),
        ("clf", make_estimator(family, best_params, seed=seed)),
    ])
    final_pipe.fit(X, y)
    return final_pipe, best_params, best_score


class BaselineModel:
    """A single-feature logistic-regression baseline (Section 5.2 'Baselines').

    Fit within the training set like any other model; kept separate from GRIDS
    because baselines are not tuned (they exist as a floor, not a competitor).
    """

    def __init__(self, feature_cols, categorical=False):
        self.feature_cols = feature_cols
        self.categorical = categorical
        self.pipe = None

    def fit(self, X, y):
        if self.categorical:
            prep = ColumnTransformer([
                ("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), self.feature_cols),
            ])
        else:
            prep = ColumnTransformer([
                ("num", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), self.feature_cols),
            ])
        self.pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])
        self.pipe.fit(X[self.feature_cols], y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[self.feature_cols])[:, 1]
