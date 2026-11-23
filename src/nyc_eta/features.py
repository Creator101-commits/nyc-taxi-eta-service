"""Request-time feature construction.

Every feature must be knowable when a rider requests a trip: zone ids,
boroughs, airport flags, centroid distances, pickup time features. Columns
like trip_distance, fares and payment type are post-trip leakage and are
rejected by ``assert_no_leakage``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from nyc_eta import config
from nyc_eta.zones import ZoneInfo

HOLIDAY_CALENDAR = USFederalHolidayCalendar()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def holiday_dates() -> set[pd.Timestamp]:
    return set(HOLIDAY_CALENDAR.holidays(start="2018-01-01", end="2023-12-31"))


class FeatureBuilder:
    """Builds the frozen feature frame from cleaned sampled trips.

    ``road_km`` comes from the OSM distance table; pairs missing from the
    table fall back to ``haversine_km * ROAD_DETOUR_FACTOR`` so the feature
    is always defined (airport zones outside the drive graph land here).
    """

    ROAD_DETOUR_FACTOR = 1.35

    def __init__(
        self,
        zone_infos: dict[int, ZoneInfo],
        road_km: dict[tuple[int, int], float] | None = None,
    ) -> None:
        self.zone_infos = zone_infos
        self.road_km = road_km or {}
        self._holidays = sorted(holiday_dates())

    def _borough(self, zone_id: int) -> str:
        info = self.zone_infos.get(zone_id)
        return info.borough if info else "Unknown"

    def _centroid(self, zone_id: int) -> tuple[float, float]:
        info = self.zone_infos.get(zone_id)
        if info is None or np.isnan(info.lat):
            return (float("nan"), float("nan"))
        return (info.lat, info.lon)

    def road_distance_km(self, pu: int, do: int) -> float:
        # Distances are directed: the drive graph is one-way aware.
        km = self.road_km.get((pu, do))
        if km is not None and np.isfinite(km):
            return float(km)
        if (do, pu) in self.road_km:
            rev = self.road_km[(do, pu)]
            if np.isfinite(rev):
                return float(rev)
        plat, plon = self._centroid(pu)
        dlat, dlon = self._centroid(do)
        if np.isnan(plat) or np.isnan(dlat):
            return float("nan")
        return haversine_km(plat, plon, dlat, dlon) * self.ROAD_DETOUR_FACTOR

    def features_for_zones(
        self, pu_zone: int, do_zone: int, pickup: pd.Timestamp
    ) -> dict[str, object]:
        """Single-request feature dict (used by the service)."""
        airport_zones = set(config.AIRPORT_ZONE_IDS.values())
        pu_lat, pu_lon = self._centroid(pu_zone)
        do_lat, do_lon = self._centroid(do_zone)
        hav = (
            haversine_km(pu_lat, pu_lon, do_lat, do_lon)
            if not (np.isnan(pu_lat) or np.isnan(do_lat))
            else float("nan")
        )
        road = self.road_distance_km(pu_zone, do_zone)
        ts = pd.Timestamp(pickup)
        return {
            "pu_zone": pu_zone,
            "do_zone": do_zone,
            "pu_borough": self._borough(pu_zone),
            "do_borough": self._borough(do_zone),
            "same_zone": int(pu_zone == do_zone),
            "pu_airport": int(pu_zone in airport_zones),
            "do_airport": int(do_zone in airport_zones),
            "haversine_km": hav,
            "road_km": road,
            "pickup_hour": int(ts.hour),
            "pickup_dow": int(ts.day_of_week),
            "is_holiday": int(pd.Timestamp(ts.date()) in self._holidays),
        }

    def build_frame(self, trips: pd.DataFrame) -> pd.DataFrame:
        """Vectorized features for a sampled trips frame.

        ``trips`` needs columns ``tpep_pickup_datetime``, ``PULocationID``,
        ``DOLocationID``. Returns the feature frame plus ``minutes``.
        """
        df = trips.copy()
        df["pickup"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["pu_zone"] = df["PULocationID"].astype(int)
        df["do_zone"] = df["DOLocationID"].astype(int)
        df["minutes"] = (
            pd.to_datetime(df["tpep_dropoff_datetime"])
            - df["pickup"]
        ).dt.total_seconds() / 60.0

        infos = self.zone_infos
        df["pu_borough"] = df["pu_zone"].map(
            lambda z: infos[z].borough if z in infos else "Unknown"
        )
        df["do_borough"] = df["do_zone"].map(
            lambda z: infos[z].borough if z in infos else "Unknown"
        )
        df["same_zone"] = (df["pu_zone"] == df["do_zone"]).astype(int)

        airport = set(config.AIRPORT_ZONE_IDS.values())
        df["pu_airport"] = df["pu_zone"].isin(airport).astype(int)
        df["do_airport"] = df["do_zone"].isin(airport).astype(int)

        lats = df["pu_zone"].map(lambda z: infos[z].lat if z in infos else np.nan)
        lons = df["pu_zone"].map(lambda z: infos[z].lon if z in infos else np.nan)
        dlats = df["do_zone"].map(lambda z: infos[z].lat if z in infos else np.nan)
        dlons = df["do_zone"].map(lambda z: infos[z].lon if z in infos else np.nan)
        df["haversine_km"] = [
            haversine_km(a, b, c, d) if not (np.isnan(a) or np.isnan(c)) else np.nan
            for a, b, c, d in zip(lats, lons, dlats, dlons, strict=False)
        ]

        df["road_km"] = [
            self.road_distance_km(int(p), int(d))
            for p, d in zip(df["pu_zone"], df["do_zone"], strict=False)
        ]

        df["pickup_hour"] = df["pickup"].dt.hour.astype(int)
        df["pickup_dow"] = df["pickup"].dt.day_of_week.astype(int)
        df["is_holiday"] = df["pickup"].dt.normalize().isin(self._holidays).astype(int)

        categorical_columns(df)
        return df


# Fixed category levels shared by every frame: LightGBM stores pandas
# category metadata at fit time and rejects predict frames whose levels
# differ, which per-month astype('category') would cause.
ZONE_ID_CATEGORIES = list(range(1, 266))
BOROUGH_CATEGORIES = [
    "Bronx", "Brooklyn", "EWR", "Manhattan", "Queens",
    "Staten Island", "Unknown",
]


def categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Assign fixed-category dtype to the protocol categorical columns."""
    zone_cols = {"pu_zone", "do_zone"}
    for col in config.CATEGORICAL_FEATURES:
        cats = ZONE_ID_CATEGORIES if col in zone_cols else BOROUGH_CATEGORIES
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


def assert_no_leakage(columns: list[str] | set[str]) -> None:
    """Fail if any forbidden post-trip column is present."""
    bad = set(columns) & config.FORBIDDEN_FEATURES
    if bad:
        raise ValueError(f"leakage columns present: {sorted(bad)}")
