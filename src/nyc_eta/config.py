"""Frozen protocol constants for the NYC taxi ETA study.

Everything a rerun must reproduce lives here: data URLs, cleaning rules,
sampling seeds, window boundaries, model grid, alarm thresholds and the
feature contract. Changing any of these changes the protocol version.
"""

from __future__ import annotations

PROTOCOL = "nyc-taxi-eta-v1"

# --- Data -----------------------------------------------------------------

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TLC_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
TLC_ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
)

# All months in the study period, inclusive.
ALL_MONTHS = [
    f"{y}-{m:02d}"
    for y in (2019, 2020, 2021)
    for m in range(1, 13)
]
TRAIN_MONTHS = ["2019-01", "2019-02", "2019-03", "2019-04"]
VALIDATION_MONTHS = ["2019-05", "2019-06"]
DEPLOY_MONTHS = [m for m in ALL_MONTHS if m >= "2019-07"]
SERVED_TRAIN_MONTHS = ["2021-10", "2021-11", "2021-12"]

# Two demo days for the website replay: a Tuesday before and after the
# COVID shock. Both are Tuesdays.
DEMO_DAYS = {"pre": "2019-10-01", "covid": "2020-04-07"}

# --- Cleaning (frozen; counts after each filter are recorded) -------------

MIN_TRIP_MINUTES = 1.0
MAX_TRIP_MINUTES = 180.0
VALID_ZONE_MIN = 1
VALID_ZONE_MAX = 263  # 264/265 are "unknown"/external and are dropped

# --- Sampling -------------------------------------------------------------

# Uniform sample of cleaned trips per month, fixed per-month seed so the
# sample is identical in every run. Full run uses SAMPLE_PER_MONTH rows;
# smoke overrides both.
SAMPLE_SEED_BASE = 20231122
SAMPLE_PER_MONTH = 200_000
SMOKE_SAMPLE_PER_MONTH = 20_000
SMOKE_MONTHS = {
    "train": ["2019-01"],
    "validation": ["2019-05"],
    "deploy": ["2019-07"],
}

# --- Features ---------------------------------------------------------------

CATEGORICAL_FEATURES = ["pu_zone", "do_zone", "pu_borough", "do_borough"]
NUMERIC_FEATURES = [
    "same_zone",
    "pu_airport",
    "do_airport",
    "haversine_km",
    "road_km",
    "pickup_hour",
    "pickup_dow",
    "is_holiday",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "minutes"  # trip duration in minutes; model predicts log1p(minutes)

AIRPORT_ZONE_IDS = {"JFK": 132, "LGA": 138, "EWR": 1}
N_ZONES = 263

# Columns that must never reach the model (post-trip leakage). The dropoff
# zone itself is allowed: it is known at request time.
FORBIDDEN_FEATURES = {
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "payment_type",
    "dropoff_datetime",
    "store_and_fwd_flag",
    "VendorID",
}

# --- Models -----------------------------------------------------------------

MODEL_GRID = [
    {"num_leaves": 63, "min_data_in_leaf": 50},
    {"num_leaves": 63, "min_data_in_leaf": 500},
    {"num_leaves": 255, "min_data_in_leaf": 50},
    {"num_leaves": 255, "min_data_in_leaf": 500},
]
LEARNING_RATE = 0.05
MAX_ROUNDS = 3000
EARLY_STOP_ROUNDS = 100
LABEL = "log1p_minutes"

# Retrain policies: at month t, labels through t-1 exist; each policy fits
# the trailing RETRAIN_WINDOW labeled months.
RETRAIN_WINDOW = 3

# --- Conformal / drift ------------------------------------------------------

CONFORMAL_ALPHA = 0.20  # 80% interval
DEGRADED_MAE_RATIO = 1.10  # degraded = MAE > 10% above validation MAE
DRIFT_SIGNALS = ["drift_share", "pred_wasserstein", "volume_ratio"]

# --- OSM routing ------------------------------------------------------------

OSM_PLACE = "New York, New York"
OSM_NETWORK_TYPE = "drive"
OSM_GRAPH_DATASET_ID = "sreeharshak101/nyc-taxi-eta-osm-graph"
