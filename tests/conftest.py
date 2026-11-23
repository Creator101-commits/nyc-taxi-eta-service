"""Shared synthetic fixtures; no network access anywhere in tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nyc_eta import config
from nyc_eta.features import FeatureBuilder
from nyc_eta.zones import ZoneInfo


def make_zone_infos(n: int = 10) -> dict[int, ZoneInfo]:
    """Tiny synthetic zone table around lower Manhattan."""
    infos: dict[int, ZoneInfo] = {}
    for i in range(1, n + 1):
        infos[i] = ZoneInfo(
            location_id=i,
            zone=f"Zone{i}",
            borough=["Manhattan", "Brooklyn", "Queens"][i % 3],
            lat=40.70 + 0.01 * i,
            lon=-74.00 - 0.01 * i,
        )
    return infos


@pytest.fixture
def zone_infos() -> dict[int, ZoneInfo]:
    return make_zone_infos()


@pytest.fixture
def builder(zone_infos) -> FeatureBuilder:
    road = {(1, 2): 3.5, (2, 1): 4.0, (1, 1): 0.0}
    return FeatureBuilder(zone_infos, road)


def make_trips(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic trips with all columns the pipeline touches."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2019-01-15 08:00")
    pu = rng.integers(1, 11, n)
    do = rng.integers(1, 11, n)
    pickup = [base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 600, n)]
    minutes = rng.uniform(3, 60, n)
    return pd.DataFrame({
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": [
            p + pd.Timedelta(minutes=float(m))
            for p, m in zip(pickup, minutes, strict=False)
        ],
        "PULocationID": pu,
        "DOLocationID": do,
    })


@pytest.fixture
def feature_frame(builder) -> pd.DataFrame:
    trips = make_trips(200, seed=1)
    return builder.build_frame(trips)


@pytest.fixture
def trip_polars() -> pl.DataFrame:
    """Polars frame exercising every cleaning filter."""
    rows = []
    # 4 good rows in-month
    for i in range(4):
        rows.append({
            "tpep_pickup_datetime": pd.Timestamp("2019-01-10 10:00")
            + pd.Timedelta(minutes=i),
            "tpep_dropoff_datetime": pd.Timestamp("2019-01-10 10:00")
            + pd.Timedelta(minutes=i + 15),
            "PULocationID": 10 + i,
            "DOLocationID": 20 + i,
        })
    # wrong month pickup
    rows.append({
        "tpep_pickup_datetime": pd.Timestamp("2019-02-01 10:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2019-02-01 10:30"),
        "PULocationID": 10, "DOLocationID": 20,
    })
    # duration too short (<1 min)
    rows.append({
        "tpep_pickup_datetime": pd.Timestamp("2019-01-10 11:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2019-01-10 11:00:30"),
        "PULocationID": 10, "DOLocationID": 20,
    })
    # duration too long (>180 min)
    rows.append({
        "tpep_pickup_datetime": pd.Timestamp("2019-01-10 12:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2019-01-10 16:00"),
        "PULocationID": 10, "DOLocationID": 20,
    })
    # invalid zones (264, 0)
    rows.append({
        "tpep_pickup_datetime": pd.Timestamp("2019-01-10 13:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2019-01-10 13:30"),
        "PULocationID": 264, "DOLocationID": 20,
    })
    rows.append({
        "tpep_pickup_datetime": pd.Timestamp("2019-01-10 14:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2019-01-10 14:30"),
        "PULocationID": 10, "DOLocationID": 0,
    })
    # dropoff before pickup but duration positive? use negative span
    rows.append({
        "tpep_pickup_datetime": pd.Timestamp("2019-01-10 15:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2019-01-10 14:00"),
        "PULocationID": 10, "DOLocationID": 20,
    })
    return pl.DataFrame(rows)


@pytest.fixture
def feature_columns() -> list[str]:
    return list(config.FEATURE_COLUMNS)
