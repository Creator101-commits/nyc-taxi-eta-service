"""Evaluation metrics: point errors, drift-alarm scores, bootstrap CIs."""

from __future__ import annotations

from typing import Any

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


def paired_day_cluster_bootstrap(
    ae_by_policy: dict[str, dict[str, np.ndarray]],
    dates_by_month: dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired day-cluster bootstrap over per-policy monthly MAE.

    ``ae_by_policy`` maps policy name -> month -> per-row absolute-error
    array aligned to that month's rows; ``dates_by_month`` maps month ->
    per-row dates aligned the same way. Each replicate draws every
    month's days with replacement once and applies the same draw to all
    policies, so policy differences are paired. A policy's statistic is
    the mean over months of the month MAE, where a replicate month MAE
    is summed drawn-day errors over summed drawn-day rows.

    ``point`` equals the plain mean of the monthly MAEs exactly, and
    diff points equal the differences of those.
    """
    months = sorted(dates_by_month)
    policies = sorted(ae_by_policy)

    point_month = {
        p: {m: float(np.mean(ae_by_policy[p][m])) for m in months}
        for p in policies
    }
    point = {
        p: float(np.mean([point_month[p][m] for m in months]))
        for p in policies
    }

    # Per-day aggregates: shared row counts, per-policy error sums.
    day_counts: dict[str, np.ndarray] = {}
    day_sums: dict[str, dict[str, np.ndarray]] = {p: {} for p in policies}
    n_days: dict[str, int] = {}
    for m in months:
        dates = np.asarray(dates_by_month[m])
        _, inv = np.unique(dates, return_inverse=True)
        day_counts[m] = np.bincount(inv).astype(np.float64)
        n_days[m] = len(day_counts[m])
        for p in policies:
            ae = np.asarray(ae_by_policy[p][m], dtype=np.float64)
            day_sums[p][m] = np.bincount(inv, weights=ae)

    rng = np.random.default_rng(seed)
    rep_policy = {p: np.full(n_boot, np.nan) for p in policies}
    rep_static_month = {m: np.full(n_boot, np.nan) for m in months}
    for b in range(n_boot):
        month_mae: dict[str, dict[str, float]] = {p: {} for p in policies}
        for m in months:
            nd = n_days[m]
            if nd == 0:
                continue
            drawn = rng.integers(0, nd, size=nd)
            denom = float(day_counts[m][drawn].sum())
            if denom <= 0:
                continue
            for p in policies:
                month_mae[p][m] = float(
                    day_sums[p][m][drawn].sum() / denom
                )
        for p in policies:
            vals = [month_mae[p][m] for m in months if m in month_mae[p]]
            if vals:
                rep_policy[p][b] = float(np.mean(vals))
        for m in months:
            if m in month_mae.get("static", {}):
                rep_static_month[m][b] = month_mae["static"][m]

    def _ci(samples: np.ndarray) -> tuple[float, float]:
        return (
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        )

    out_policies: dict[str, dict[str, float]] = {}
    for p in policies:
        lo, hi = _ci(rep_policy[p])
        out_policies[p] = {
            "point": point[p],
            "boot_mean": float(np.nanmean(rep_policy[p])),
            "ci_low": lo,
            "ci_high": hi,
        }

    diff_pairs = {
        "monthly-static": ("monthly", "static"),
        "drift-static": ("drift", "static"),
        "drift-monthly": ("drift", "monthly"),
    }
    diffs: dict[str, dict[str, float]] = {}
    for name, (pa, pb) in diff_pairs.items():
        if pa not in rep_policy or pb not in rep_policy:
            continue
        samples = rep_policy[pa] - rep_policy[pb]
        lo, hi = _ci(samples)
        diffs[name] = {
            "point": point[pa] - point[pb],
            "boot_mean": float(np.nanmean(samples)),
            "ci_low": lo,
            "ci_high": hi,
            "share_below_zero": float(np.nanmean(samples < 0)),
        }

    static_monthly_ci = {
        m: {
            "point": point_month["static"][m],
            "ci_low": _ci(rep_static_month[m])[0],
            "ci_high": _ci(rep_static_month[m])[1],
        }
        for m in months
        if "static" in point_month
    }

    return {
        "method": (
            "days resampled within each month, same draw for all "
            "policies; conditional on the trained models"
        ),
        "n_boot": n_boot,
        "seed": seed,
        "policies": out_policies,
        "diffs": diffs,
        "static_monthly_ci": static_monthly_ci,
    }
