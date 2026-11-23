"""Evaluation metrics: point errors, drift-alarm scores, bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute percentage error; true values are minutes >= 1."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = yt > 0
    if not np.any(mask):
        return float("nan")
    return float(np.median(np.abs(yt[mask] - yp[mask]) / yt[mask]))


def point_metrics(
    df: pd.DataFrame, pred_minutes: np.ndarray
) -> dict[str, float]:
    y = df["minutes"].to_numpy()
    p = np.asarray(pred_minutes, dtype=np.float64)
    return {
        "mae": mae(y, p),
        "rmse": rmse(y, p),
        "median_ape": median_ape(y, p),
        "n": int(len(y)),
    }


def borough_slices(
    df: pd.DataFrame, pred_minutes: np.ndarray, by: str = "pu_borough"
) -> dict[str, dict[str, float]]:
    """MAE/RMSE per borough slice (default: pickup borough)."""
    out: dict[str, dict[str, float]] = {}
    pred = pd.Series(np.asarray(pred_minutes), index=df.index)
    for borough, idx in df.groupby(by, observed=True).groups.items():
        sub = df.loc[idx]
        p = pred.loc[idx].to_numpy()
        out[str(borough)] = {
            "mae": mae(sub["minutes"].to_numpy(), p),
            "n": int(len(sub)),
        }
    return out


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation; NaN when a series is constant."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rho, _ = stats.spearmanr(x, y)
    return float(rho)


def alarm_scores(
    degraded: list[bool], alarms: list[bool]
) -> dict[str, float]:
    """Precision/recall of alarms for degraded months."""
    tp = sum(1 for d, a in zip(degraded, alarms, strict=False) if d and a)
    fp = sum(1 for d, a in zip(degraded, alarms, strict=False) if not d and a)
    fn = sum(1 for d, a in zip(degraded, alarms, strict=False) if d and not a)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall}


def day_cluster_bootstrap(
    frames: dict[str, pd.DataFrame],
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI for mean monthly MAE, resampling days (clusters).

    ``frames`` maps month -> DataFrame with columns ``date`` and ``ae``
    (absolute error per row). Each resample draws dates with replacement
    inside every month, recomputes that month's MAE over all rows of the
    drawn dates, then averages the monthly MAEs.
    """
    rng = np.random.default_rng(seed)
    months = sorted(frames)
    # Pre-group rows by date per month.
    by_month: dict[str, dict[object, np.ndarray]] = {}
    for m in months:
        df = frames[m]
        by_month[m] = {
            d: g["ae"].to_numpy()
            for d, g in df.groupby("date", sort=False)
        }
    stats_out = []
    for _ in range(n_boot):
        month_maes = []
        for m in months:
            days = list(by_month[m])
            if not days:
                continue
            drawn = rng.choice(len(days), size=len(days), replace=True)
            errs = np.concatenate([by_month[m][days[i]] for i in drawn])
            month_maes.append(float(np.mean(errs)))
        if month_maes:
            stats_out.append(float(np.mean(month_maes)))
    arr = np.asarray(stats_out)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_boot": n_boot,
    }
