"""Discrimination and calibration metrics (Section 7.1).

AUC, average precision for discrimination; Brier score, calibration-in-the-large,
calibration slope and expected calibration error (ECE) for calibration, following
Van Calster et al. (2019).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_ap(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def brier(y_true, y_prob):
    return brier_score_loss(y_true, y_prob)


def calibration_in_the_large(y_true, y_prob):
    """Observed event rate minus mean predicted risk (0 = perfectly calibrated in the large)."""
    return float(np.mean(y_true) - np.mean(y_prob))


def calibration_slope(y_true, y_prob, eps=1e-6):
    """Slope of logistic regression of y on logit(p); 1 = ideal calibration."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return np.nan
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p, y)
    return float(lr.coef_[0][0])


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def all_metrics(y_true, y_prob):
    return {
        "auc": safe_auc(y_true, y_prob),
        "ap": safe_ap(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "cal_in_large": calibration_in_the_large(y_true, y_prob),
        "cal_slope": calibration_slope(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": len(y_true),
        "prevalence": float(np.mean(y_true)),
    }
