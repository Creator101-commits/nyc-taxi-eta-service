"""Retraining policy simulation.

Timing contract (frozen): at deployment month t, labels exist through
t-1. The monthly policy fits months t-3..t-1. The drift policy fits the
same window only when the alarm fired on month t-1's label-free signals.
The static policy never retrains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nyc_eta import config
from nyc_eta.drift import DriftSignals

# A "fit" callable receives the concatenated labeled training frame and
# returns a predictor object with .predict(df) -> minutes array.
FitFn = Callable[[pd.DataFrame], object]
PredictFn = Callable[[object, pd.DataFrame], np.ndarray]


@dataclass
class MonthData:
    """Per-month evaluation payload."""

    month: str
    features: pd.DataFrame
    minutes: np.ndarray
    signals: DriftSignals | None = None
    alarm_fired_prev: bool = False


@dataclass
class PolicyResult:
    name: str
    monthly_mae: dict[str, float] = field(default_factory=dict)
    # Per-row absolute errors per month, aligned to the month's rows;
    # feeds the paired day-cluster bootstrap.
    monthly_ae: dict[str, np.ndarray] = field(default_factory=dict)
    n_retrains: int = 0
    retrain_months: list[str] = field(default_factory=list)


def _window_months(months: list[str], t: str) -> list[str]:
    """The RETRAIN_WINDOW labeled months ending at t-1 (exclusive of t)."""
    idx = months.index(t)
    start = max(0, idx - config.RETRAIN_WINDOW)
    return months[start:idx]


def simulate_policies(
    deploy_months: list[str],
    month_data: dict[str, MonthData],
    labeled_pool: dict[str, pd.DataFrame],
    fit_fn: FitFn,
    predict_fn: PredictFn,
    static_model: object,
    signal_thresholds: dict[str, float],
) -> dict[str, PolicyResult]:
    """Simulate static / monthly / drift policies over deployment months.

    ``labeled_pool`` maps month -> feature frame for months whose labels are
    available (train + validation + already-deployed months as time passes).
    """
    results = {
        "static": PolicyResult("static"),
        "monthly": PolicyResult("monthly"),
        "drift": PolicyResult("drift"),
    }
    monthly_model = static_model
    drift_model = static_model
    labeled_available = dict(labeled_pool)

    ordered = [m for m in deploy_months if m in month_data]
    for t in ordered:
        data = month_data[t]
        y = data.minutes
        feats = data.features

        # static
        p_static = predict_fn(static_model, feats)
        ae_static = np.abs(y - p_static)
        results["static"].monthly_ae[t] = ae_static
        results["static"].monthly_mae[t] = float(np.mean(ae_static))

        # Trailing labeled months immediately before t (up to RETRAIN_WINDOW).
        avail = sorted(m for m in labeled_available if m < t)
        win = avail[-config.RETRAIN_WINDOW:]

        # monthly: retrain on the trailing window when any labels exist
        if win:
            train_df = pd.concat([labeled_available[m] for m in win],
                                 ignore_index=True)
            monthly_model = fit_fn(train_df)
            results["monthly"].n_retrains += 1
            results["monthly"].retrain_months.append(t)
        p_monthly = predict_fn(monthly_model, feats)
        ae_monthly = np.abs(y - p_monthly)
        results["monthly"].monthly_ae[t] = ae_monthly
        results["monthly"].monthly_mae[t] = float(np.mean(ae_monthly))

        # drift: retrain only when the previous deploy month's label-free
        # signals alarmed (month t-1's signals vs frozen thresholds)
        prev = ordered[ordered.index(t) - 1] if ordered.index(t) > 0 else None
        alarm_prev = False
        if prev is not None and win:
            prev_signals = month_data[prev].signals
            if prev_signals is not None:
                from nyc_eta.drift import any_alarm, signal_alarms  # noqa: PLC0415

                alarm_prev = any_alarm(
                    signal_alarms(prev_signals, signal_thresholds)
                )
        if alarm_prev:
            train_df = pd.concat([labeled_available[m] for m in win],
                                 ignore_index=True)
            drift_model = fit_fn(train_df)
            results["drift"].n_retrains += 1
            results["drift"].retrain_months.append(t)
        p_drift = predict_fn(drift_model, feats)
        ae_drift = np.abs(y - p_drift)
        results["drift"].monthly_ae[t] = ae_drift
        results["drift"].monthly_mae[t] = float(np.mean(ae_drift))

        # month t's labels become available for later windows
        labeled_available[t] = feats.assign(minutes=y)

    return results
