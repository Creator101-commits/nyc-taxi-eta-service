"""Tests for models, conformal intervals, drift signals and policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyc_eta import config, drift, models, policies
from nyc_eta.conformal import (
    ConformalInterval,
    empirical_coverage,
    fit_conformal,
    predict_interval,
)


class TestBaseline:
    def test_pair_median_with_fallbacks(self, feature_frame):
        base = models.ZonePairBaseline().fit(feature_frame)
        # a seen pair predicts its median
        row = feature_frame.iloc[[0]]
        pred = base.predict(row)
        key = (int(row["pu_zone"].iloc[0]), int(row["do_zone"].iloc[0]))
        assert pred[0] == pytest.approx(base.pair_medians[key])

    def test_unknown_pair_uses_borough_or_global(self, feature_frame):
        base = models.ZonePairBaseline().fit(feature_frame)
        fake = feature_frame.iloc[[0]].copy()
        fake["pu_zone"] = 999  # unseen pair
        fake["pu_borough"] = "Nowhere"
        pred = base.predict(fake)
        assert pred[0] == pytest.approx(base.global_median)


class TestLGBM:
    def test_tiny_train_and_predict(self, feature_frame):
        train = feature_frame.iloc[:150]
        val = feature_frame.iloc[150:]
        cfg = models.LGBMConfig(num_leaves=15, min_data_in_leaf=10)
        model = models.train_lgbm(train, val, cfg)
        assert model.best_iteration >= 1
        pred = models.predict_minutes(model, val)
        assert pred.shape == (len(val),)
        assert np.all(pred >= 0)

    def test_fixed_iterations(self, feature_frame):
        cfg = models.LGBMConfig(num_leaves=15, min_data_in_leaf=10)
        model = models.train_lgbm_fixed(
            feature_frame.iloc[:150], cfg, iterations=7
        )
        assert model.best_iteration == 7
        assert model.booster.num_trees() == 7

    def test_select_model_records(self, feature_frame):
        train = feature_frame.iloc[:150]
        val = feature_frame.iloc[150:]
        best, records = models.select_model(train, val)
        assert len(records) == len(config.MODEL_GRID)
        assert all("best_iteration" in r and "cap_bound" in r
                   for r in records)
        assert best.val_mae == min(r["val_mae"] for r in records)


class TestConformal:
    def test_interval_covers_nominal(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(5, 45, 500)
        p = y + rng.normal(0, 3, 500)
        p = np.clip(p, 0.5, None)
        interval = fit_conformal(y, p, alpha=0.2)
        lo, hi = predict_interval(np.log1p(np.clip(p, 0, None)), interval)
        coverage = np.mean((y >= lo) & (y <= hi))
        assert coverage >= 0.7  # generous bound for a small sample

    def test_empirical_coverage(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        p = np.array([10.5, 19.0, 31.0, 38.0])
        interval = ConformalInterval(q=0.5, alpha=0.2)
        cov = empirical_coverage(y, p, interval)
        assert 0.0 <= cov <= 1.0


class TestDrift:
    def test_wasserstein_and_volume(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert drift.pred_wasserstein_distance(a, b) == pytest.approx(1.0)
        assert drift.volume_ratio(100, 50.0) == pytest.approx(2.0)

    def test_volume_ratio_uses_cleaned_counts(self):
        # Equal per-month samples (both capped at sample_n) but different
        # cleaned totals: len(sampled_df) would give identical ratios,
        # so the signal must be fed the final-filter counts.
        counts = {
            "2019-01": {"dropoff_after_pickup": 7_400_000},
            "2019-02": {"dropoff_after_pickup": 7_300_000},
            "2020-04": {"dropoff_after_pickup": 2_900_000},
        }
        n = drift.cleaned_counts_for_months(counts, list(counts))
        ref = float(np.mean([n["2019-01"], n["2019-02"]]))
        covid = drift.volume_ratio(n["2020-04"], ref)
        normal = drift.volume_ratio(n["2019-01"], ref)
        assert covid != pytest.approx(normal)
        assert covid == pytest.approx(2_900_000 / 7_350_000)

    def test_thresholds_and_alarms(self):
        sigs = [
            drift.DriftSignals(0.1, 1.0, 1.0),
            drift.DriftSignals(0.2, 2.0, 1.5),
        ]
        th = drift.alarm_thresholds(sigs)
        assert th["drift_share"] == pytest.approx(0.2)
        assert th["pred_wasserstein"] == pytest.approx(2.0)
        cur = drift.DriftSignals(0.5, 3.0, 1.0)
        alarms = drift.signal_alarms(cur, th)
        assert alarms["drift_share"] and alarms["pred_wasserstein"]
        assert not alarms["volume_ratio"]
        assert drift.any_alarm(alarms)


class TestPolicies:
    def _data(self):
        rng = np.random.default_rng(0)
        months = ["2019-01", "2019-02", "2019-05", "2019-07", "2019-08"]
        md: dict[str, policies.MonthData] = {}
        for m in months:
            n = 20
            feats = pd.DataFrame({
                "minutes": rng.uniform(5, 40, n),
                "x": rng.normal(size=n),
            })
            md[m] = policies.MonthData(
                month=m, features=feats,
                minutes=feats["minutes"].to_numpy(),
            )
        return months, md

    def test_static_never_retrains(self):
        months, md = self._data()
        deploy = ["2019-07", "2019-08"]
        pool = {m: md[m] for m in ["2019-01", "2019-02", "2019-05"]}
        labeled = {
            m: pool[m].features.assign(minutes=pool[m].minutes)
            for m in pool
        }
        results = policies.simulate_policies(
            deploy, {m: md[m] for m in deploy}, labeled,
            fit_fn=lambda df: object(),
            predict_fn=lambda model, df: df["minutes"].to_numpy() * 0 + 10.0,
            static_model=object(),
            signal_thresholds={},
        )
        assert results["static"].n_retrains == 0
        assert results["monthly"].n_retrains == 2
        assert len(results["drift"].monthly_mae) == 2

    def test_monthly_uses_trailing_window(self):
        # fit_fn records which months it saw via the minutes column range
        seen: list[set] = []

        def fit_fn(df):
            seen.append(set(df.index))
            return object()

        months, md = self._data()
        deploy = ["2019-07", "2019-08"]
        labeled = {
            m: md[m].features.assign(minutes=md[m].minutes)
            for m in ["2019-01", "2019-02", "2019-05"]
        }
        policies.simulate_policies(
            deploy, {m: md[m] for m in deploy}, labeled,
            fit_fn=fit_fn,
            predict_fn=lambda model, df: np.full(len(df), 10.0),
            static_model=object(),
            signal_thresholds={},
        )
        # two deploy months -> two monthly retrains
        assert len(seen) == 2

    def test_drift_retrains_only_on_alarm(self):
        calls = {"n": 0}

        def fit_fn(df):
            calls["n"] += 1
            return object()

        months, md = self._data()
        deploy = ["2019-07", "2019-08"]
        # month_data needs signals for the drift check on previous month
        md["2019-07"].signals = drift.DriftSignals(0.9, 9.0, 9.0)
        md["2019-08"].signals = drift.DriftSignals(0.0, 0.0, 0.0)
        labeled = {
            m: md[m].features.assign(minutes=md[m].minutes)
            for m in ["2019-01", "2019-02", "2019-05"]
        }
        thresholds = {"drift_share": 0.5, "pred_wasserstein": 5.0,
                      "volume_ratio": 5.0}
        results = policies.simulate_policies(
            deploy, {m: md[m] for m in deploy}, labeled,
            fit_fn=fit_fn,
            predict_fn=lambda model, df: np.full(len(df), 10.0),
            static_model=object(),
            signal_thresholds=thresholds,
        )
        # monthly retrains twice; drift retrains once (alarm on 2019-07
        # signals triggers the 2019-08 retrain)
        assert results["monthly"].n_retrains == 2
        assert results["drift"].n_retrains == 1
        assert results["drift"].retrain_months == ["2019-08"]
