"""Label-free monthly drift signals and alarm logic.

Signals (all computable without labels):
  - drift_share: share of features flagged drifted by Evidently
    DataDriftPreset vs the train reference sample.
  - pred_wasserstein: Wasserstein distance between reference and month
    prediction distributions (minutes scale).
  - volume_ratio: month sample size / mean monthly train sample size.

Alarm threshold per signal = max over the validation months' values
(pre-specified). Degraded month = MAE > 10% above validation MAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from nyc_eta import config


@dataclass(frozen=True)
class DriftSignals:
    drift_share: float
    pred_wasserstein: float
    volume_ratio: float


def pred_wasserstein_distance(
    ref_pred: np.ndarray, cur_pred: np.ndarray
) -> float:
    return float(wasserstein_distance(ref_pred, cur_pred))


def volume_ratio(cur_n: int, ref_monthly_mean: float) -> float:
    if ref_monthly_mean <= 0:
        return float("nan")
    return float(cur_n / ref_monthly_mean)


def drift_share(
    ref_df: pd.DataFrame, cur_df: pd.DataFrame
) -> float:
    """Evidently DataDriftPreset drifted-feature share (lazy import).

    evidently 0.7.x API: Report([preset]).run(current, reference) returns
    a Snapshot; the DriftedColumnsCount metric carries count + share.
    """
    from evidently import Report  # noqa: PLC0415 - kernel-only dep
    from evidently.presets import DataDriftPreset  # noqa: PLC0415

    feats = config.FEATURE_COLUMNS
    report = Report([DataDriftPreset()])
    snapshot = report.run(cur_df[feats], ref_df[feats])
    for metric in snapshot.dict().get("metrics", []):
        if metric.get("config", {}).get("type") == (
            "evidently:metric_v2:DriftedColumnsCount"
        ):
            return float(metric["value"]["share"])
    return float("nan")


def compute_signals(
    ref_features: pd.DataFrame,
    ref_pred: np.ndarray,
    ref_monthly_mean: float,
    cur_features: pd.DataFrame,
    cur_pred: np.ndarray,
    cur_n: int,
) -> DriftSignals:
    return DriftSignals(
        drift_share=drift_share(ref_features, cur_features),
        pred_wasserstein=pred_wasserstein_distance(ref_pred, cur_pred),
        volume_ratio=volume_ratio(cur_n, ref_monthly_mean),
    )


def alarm_thresholds(val_signals: list[DriftSignals]) -> dict[str, float]:
    """Per-signal threshold = max over validation months (pre-specified)."""
    out: dict[str, float] = {}
    for name in config.DRIFT_SIGNALS:
        vals = [getattr(s, name) for s in val_signals]
        finite = [v for v in vals if np.isfinite(v)]
        out[name] = max(finite) if finite else float("nan")
    return out


def signal_alarms(signals: DriftSignals, thresholds: dict[str, float]) -> dict[str, bool]:
    return {
        name: bool(np.isfinite(getattr(signals, name))
                   and np.isfinite(thresholds.get(name, float("nan")))
                   and getattr(signals, name) > thresholds[name])
        for name in config.DRIFT_SIGNALS
    }


def any_alarm(signal_alarms_map: dict[str, bool]) -> bool:
    return any(signal_alarms_map.values())
