"""Tests for TLC cleaning counts, sampling determinism, features, metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_trips as _trips

from nyc_eta import config, data, drift, experiments, metrics, models, policies
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


def _day_trips(day: str, n: int, seed: int, pu: int, do: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(day)
    pickup = [
        base + pd.Timedelta(minutes=int(m))
        for m in rng.integers(0, 600, n)
    ]
    minutes = rng.uniform(5, 40, n)
    return pd.DataFrame({
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": [
            p + pd.Timedelta(minutes=float(m))
            for p, m in zip(pickup, minutes, strict=False)
        ],
        "PULocationID": np.full(n, pu),
        "DOLocationID": np.full(n, do),
    })


class TestWebExport:
    def _frames(self, builder):
        pre = builder.build_frame(_day_trips("2019-10-01", 100, 1, 1, 2))
        covid = builder.build_frame(_day_trips("2020-04-07", 200, 2, 2, 1))
        return {"2019-10": pre, "2020-04": covid}

    def _call(self, tmp_path, builder, zone_infos):
        feats = self._frames(builder)
        static = models.ZonePairBaseline().fit(
            pd.concat(list(feats.values()), ignore_index=True)
        )
        pol = {
            name: policies.PolicyResult(
                name=name,
                monthly_mae={"2019-10": 3.5},
                retrain_months=["2019-10"] if name != "static" else [],
            )
            for name in ("static", "monthly", "drift")
        }
        calls: list[set[tuple[int, int]]] = []

        def stub_routes(pairs):
            calls.append(set(pairs))
            out = {}
            for a, b in pairs:
                if (a, b) == (2, 1):
                    out[f"{a}>{b}"] = [[-74.0, 40.7]]  # degenerate path
                else:
                    out[f"{a}>{b}"] = [
                        [-74.0, 40.7], [-73.9, 40.8], [-73.8, 40.9],
                    ]
            return out

        zones_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-74.1, 40.6], [-73.9, 40.6],
                        [-73.9, 40.8], [-74.1, 40.8], [-74.1, 40.6],
                    ]],
                },
            }],
        }
        meta = experiments._export_web_data(
            tmp_path, zones_geojson, feats,
            {"2019-10": drift.DriftSignals(0.1, 0.2, 1.0)},
            {"2019-10": {"mae": 3.5, "degraded": False, "coverage": 0.8}},
            ["2019-10"], None, zone_infos,
            static_model=static, pol_results=pol,
            thresholds={
                "drift_share": 0.5,
                "pred_wasserstein": 0.5,
                "volume_ratio": 2.0,
            },
            val_mae=4.0, clean_counts={
                "2019-10": {"dropoff_after_pickup": 1_000_000},
                "2020-04": {"dropoff_after_pickup": 250_000},
            },
            sample_n=1000, route_fn=stub_routes,
        )
        return tmp_path / "web", meta, calls

    def test_proportional_counts_and_pair_filter(
        self, tmp_path, builder, zone_infos
    ):
        web, meta, calls = self._call(tmp_path, builder, zone_infos)
        trips = json.loads((web / "trips.json").read_text())
        days = trips["days"]
        # est_real = n_sampled_day * cleaned_n_month / sample_n_month
        assert days["pre"]["est_real_trips"] == 100_000
        assert days["pre"]["n_exported"] == 100
        assert days["pre"]["real_trips_per_taxi"] == pytest.approx(1000.0)
        assert days["covid"]["est_real_trips"] == 50_000
        # covid exports round(100 * 50000/100000) = 50 trips
        assert days["covid"]["n_exported"] == 50
        assert days["covid"]["real_trips_per_taxi"] == pytest.approx(1000.0)
        assert len(trips["trips"]["covid"]) == 50
        row = trips["trips"]["pre"][0]
        assert set(row) == {"pu", "do", "minute", "minutes", "pred_minutes"}
        assert row["minutes"] > 0 and row["pred_minutes"] > 0

        # only exported pairs reach the router and routes.json
        assert calls == [{(1, 2), (2, 1)}]
        routes = json.loads((web / "routes.json").read_text())
        assert set(routes) == {"1>2", "2>1"}
        # degenerate one-point routes pass through rounded, unsimplified
        assert routes["2>1"] == [[-74.0, 40.7]]
        assert len(routes["1>2"]) >= 2

        drift_json = json.loads((web / "drift.json").read_text())
        assert drift_json["val_mae"] == 4.0
        assert drift_json["degraded_cut"] == pytest.approx(4.4)
        assert drift_json["alarms"]["2019-10"]["any"] is False
        assert set(drift_json["policies"]) == {"static", "monthly", "drift"}

        assert (web / "zones.geojson").exists()
        assert meta["caps"]["pre_day_trips"] == config.WEB_TRIPS_PRE_CAP
        assert meta["seeds"]["subset"] == config.WEB_SUBSET_SEED
        assert set(meta["files"]) == {
            "zones.geojson", "drift.json", "trips.json", "routes.json",
        }
        assert all(v > 0 for v in meta["files"].values())


def _dense_ring(x0: float, y0: float, size: float = 0.002,
                step: float = 0.00002) -> list[list[float]]:
    """Square ring with collinear points every ``step`` degrees."""
    per = int(round(size / step))
    pts: list[list[float]] = []
    for i in range(per + 1):
        pts.append([x0 + i * step, y0])
    for i in range(1, per + 1):
        pts.append([x0 + size, y0 + i * step])
    for i in range(1, per + 1):
        pts.append([x0 + size - i * step, y0 + size])
    for i in range(1, per):
        pts.append([x0, y0 + size - i * step])
    pts.append([x0, y0])
    return pts


class TestZoneGeojsonSimplify:
    def test_polygon_and_multipolygon_simplified_and_rounded(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_dense_ring(-74.123456789, 40.6)],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [
                            [_dense_ring(-74.09, 40.65)],
                            [_dense_ring(-74.04, 40.70)],
                        ],
                    },
                },
            ],
        }
        experiments._simplify_zones_geojson(fc)

        poly_ring = fc["features"][0]["geometry"]["coordinates"][0]
        mp = fc["features"][1]["geometry"]["coordinates"]
        # ~400-point rings collapse to a handful of corners.
        assert len(poly_ring) < 20
        assert len(mp) == 2
        assert all(len(part[0]) < 20 for part in mp)

        def walk(c: object) -> None:
            seq = c  # type: ignore[assignment]
            if isinstance(seq[0], (int, float)):
                assert seq[0] == round(seq[0], 5)
                assert seq[1] == round(seq[1], 5)
            else:
                for e in seq:
                    walk(e)

        for feat in fc["features"]:
            walk(feat["geometry"]["coordinates"])
