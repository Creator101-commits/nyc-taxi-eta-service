"""Tests for TLC cleaning counts, sampling determinism, features, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_trips as _trips

from nyc_eta import config, data, metrics
from nyc_eta.features import (
    BOROUGH_CATEGORIES,
    ZONE_ID_CATEGORIES,
    assert_no_leakage,
)


class TestCleaning:
    def test_counts_after_each_filter(self, trip_polars):
        _, counts = data.clean_trips(trip_polars, "2019-01")
        assert counts["raw"] == 10
        # pickup_in_month drops the Feb row -> 9
        assert counts["pickup_in_month"] == 9
        # duration_ok drops <1min and >180min -> 6
        assert counts["duration_ok"] == 6
        # zones_ok drops 264 and 0 -> 4
        assert counts["zones_ok"] == 4
        # dropoff_after_pickup: the negative-span row already failed
        # duration (negative minutes < 1) so count stays 4
        assert counts["dropoff_after_pickup"] == 4

    def test_dropoff_before_pickup_filtered(self):
        import polars as pl

        df = pl.DataFrame({
            "tpep_pickup_datetime": [pd.Timestamp("2019-01-10 15:00")],
            "tpep_dropoff_datetime": [pd.Timestamp("2019-01-10 14:00")],
            "PULocationID": [10],
            "DOLocationID": [20],
        })
        # negative duration fails the [1,180] filter first
        cleaned, counts = data.clean_trips(df, "2019-01")
        assert cleaned.height == 0
        assert counts["duration_ok"] == 0

    def test_sampling_deterministic(self, trip_polars):
        cleaned, _ = data.clean_trips(trip_polars, "2019-01")
        s1 = data.sample_trips(cleaned, "2019-01", 3)
        s2 = data.sample_trips(cleaned, "2019-01", 3)
        assert s1.equals(s2)

    def test_sampling_capped_at_height(self, trip_polars):
        cleaned, _ = data.clean_trips(trip_polars, "2019-01")
        s = data.sample_trips(cleaned, "2019-01", 100)
        assert s.height == cleaned.height


class TestFeatures:
    def test_build_frame_columns(self, feature_frame):
        for col in config.FEATURE_COLUMNS:
            assert col in feature_frame.columns
        assert "minutes" in feature_frame.columns
        assert len(feature_frame) == 200

    def test_no_leakage_columns(self, feature_frame):
        assert_no_leakage(list(feature_frame.columns))

    def test_leakage_guard_rejects(self):
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage(["pu_zone", "fare_amount", "trip_distance"])

    def test_categorical_dtypes(self, feature_frame):
        for col in config.CATEGORICAL_FEATURES:
            assert str(feature_frame[col].dtype) == "category"

    def test_categorical_levels_fixed_across_frames(self, builder):
        # LightGBM requires identical pandas category metadata at predict
        # time; per-frame astype('category') produced different levels.
        f1 = builder.build_frame(_trips(50, seed=3))
        f2 = builder.build_frame(_trips(50, seed=4))
        for col in config.CATEGORICAL_FEATURES:
            expected = (
                ZONE_ID_CATEGORIES
                if col.endswith("_zone")
                else BOROUGH_CATEGORIES
            )
            assert list(f1[col].cat.categories) == expected
            assert list(f2[col].cat.categories) == expected

    def test_road_distance_directed(self, builder, zone_infos):
        # (1,2)=3.5 directed, (2,1)=4.0 directed (not symmetrized)
        assert builder.road_distance_km(1, 2) == 3.5
        assert builder.road_distance_km(2, 1) == 4.0

    def test_road_distance_fallback(self, builder):
        # unknown pair falls back to haversine * detour factor
        km = builder.road_distance_km(5, 9)
        assert km > 0 and np.isfinite(km)

    def test_features_for_zones_request_time(self, builder):
        feats = builder.features_for_zones(1, 2, pd.Timestamp("2019-07-04"))
        assert feats["pu_zone"] == 1 and feats["do_zone"] == 2
        assert feats["road_km"] == 3.5
        assert feats["is_holiday"] == 1  # July 4 federal holiday
        assert feats["pickup_hour"] == 0


class TestMetrics:
    def test_mae_rmse(self):
        y = np.array([10.0, 20.0, 30.0])
        p = np.array([12.0, 18.0, 30.0])
        assert metrics.mae(y, p) == pytest.approx(4 / 3)
        assert metrics.rmse(y, p) == pytest.approx(np.sqrt(8 / 3))

    def test_median_ape(self):
        y = np.array([10.0, 20.0, 40.0])
        p = np.array([11.0, 20.0, 44.0])
        assert metrics.median_ape(y, p) == pytest.approx(0.1)

    def test_spearman_constant_nan(self):
        assert np.isnan(metrics.spearman([1, 1, 1], [1, 2, 3]))

    def test_alarm_scores(self):
        s = metrics.alarm_scores(
            degraded=[True, False, True],
            alarms=[True, True, False],
        )
        assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 1
        assert s["precision"] == pytest.approx(0.5)
        assert s["recall"] == pytest.approx(0.5)

    def test_borough_slices(self, feature_frame):
        pred = np.full(len(feature_frame), 15.0)
        slices = metrics.borough_slices(feature_frame, pred)
        assert set(slices) == set(feature_frame["pu_borough"].unique())
        assert all("mae" in v and "n" in v for v in slices.values())
