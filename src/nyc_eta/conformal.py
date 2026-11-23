"""Split-conformal intervals on the log1p scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nyc_eta.models import expm1_pred, log1p_target


@dataclass(frozen=True)
class ConformalInterval:
    """Half-width q in log1p space; minutes interval = expm1(log_pred +/- q)."""

    q: float
    alpha: float


def fit_conformal(
    val_minutes: np.ndarray, val_pred_minutes: np.ndarray, alpha: float
) -> ConformalInterval:
    """Split-conformal quantile of absolute log-scale residuals."""
    y_log = log1p_target(val_minutes)
    p_log = log1p_target(np.clip(val_pred_minutes, 0.0, None))
    residuals = np.abs(y_log - p_log)
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(residuals, level, method="higher"))
    return ConformalInterval(q=q, alpha=alpha)


def predict_interval(
    pred_log: np.ndarray, interval: ConformalInterval
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower_minutes, upper_minutes) arrays."""
    lo_log = pred_log - interval.q
    hi_log = pred_log + interval.q
    return expm1_pred(lo_log), expm1_pred(hi_log)


def empirical_coverage(
    y_minutes: np.ndarray,
    pred_minutes: np.ndarray,
    interval: ConformalInterval,
) -> float:
    """Fraction of observations inside the conformal interval."""
    pred_log = log1p_target(np.clip(pred_minutes, 0.0, None))
    lo, hi = predict_interval(pred_log, interval)
    inside = (y_minutes >= lo) & (y_minutes <= hi)
    return float(np.mean(inside))
