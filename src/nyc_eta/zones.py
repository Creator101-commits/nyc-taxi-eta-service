"""Taxi zone metadata: lookup table, polygons, centroids, GeoJSON.

Sources (pinned by sha256 in the run manifest):
  - taxi_zone_lookup.csv: LocationID -> Zone, Borough
  - taxi_zones.zip: shapefile with one polygon per LocationID
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZoneInfo:
    location_id: int
    zone: str
    borough: str
    lat: float
    lon: float


def load_zone_lookup(csv_path: Path) -> pd.DataFrame:
    """taxi_zone_lookup.csv -> DataFrame indexed by LocationID."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"LocationID": "location_id", "Zone": "zone",
                            "Borough": "borough"})
    return df.set_index("location_id")


def _polygon_centroid(rings: list[list[tuple[float, float]]]) -> tuple[float, float]:
    """Area centroid of the largest ring set; falls back to vertex mean."""
    best: list[tuple[float, float]] | None = None
    best_area = -1.0
    for ring in rings:
        if len(ring) < 3:
            continue
        area = 0.0
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=False):
            area += x1 * y2 - x2 * y1
        area = abs(area) / 2.0
        if area > best_area:
            best_area = area
            best = ring
    if best is None:
        pts = [p for ring in rings for p in ring]
        return (float(np.mean([p[1] for p in pts])),
                float(np.mean([p[0] for p in pts])))
    a = c_x = c_y = 0.0
    for (x1, y1), (x2, y2) in zip(best, best[1:] + best[:1], strict=False):
        cross = x1 * y2 - x2 * y1
        a += cross
        c_x += (x1 + x2) * cross
        c_y += (y1 + y2) * cross
    if abs(a) < 1e-12:
        return (float(np.mean([p[1] for p in best])),
                float(np.mean([p[0] for p in best])))
    a *= 0.5
    return (c_y / (6.0 * a), c_x / (6.0 * a))


def _wgs84_transformer(prj_wkt: str) -> Any:
    """pyproj transformer from the shapefile CRS to EPSG:4326.

    The TLC taxi_zones shapefile is projected (NAD83 StatePlane NY Long
    Island, US survey feet); without this step centroids and GeoJSON
    coordinates are meaningless to consumers expecting lon/lat.
    """
    from pyproj import Transformer  # noqa: PLC0415

    if not prj_wkt.strip():
        return None
    return Transformer.from_crs(prj_wkt, "EPSG:4326", always_xy=True)


def load_zone_shapes(zip_path: Path) -> dict[int, dict[str, Any]]:
    """Parse taxi_zones.zip shapefile -> {location_id: geojson-ish polygon}.

    Rings are returned as (lon, lat) pairs in WGS84, reprojected from the
    shapefile's .prj CRS when present.
    """
    import shapefile  # pyshp, pure python

    with zipfile.ZipFile(zip_path) as zf:
        names = {n.lower(): n for n in zf.namelist()}
        shp_name = next(n for n in names if n.endswith(".shp"))
        dbf_name = next(n for n in names if n.endswith(".dbf"))
        prj_name = next((n for n in names if n.endswith(".prj")), None)
        shp_bytes = io.BytesIO(zf.read(names[shp_name]))
        dbf_bytes = io.BytesIO(zf.read(names[dbf_name]))
        prj_wkt = (
            zf.read(names[prj_name]).decode() if prj_name is not None else ""
        )

    to_wgs84 = _wgs84_transformer(prj_wkt)

    def _ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        ring = [(float(p[0]), float(p[1])) for p in points]
        if to_wgs84 is None or not ring:
            return ring
        lons, lats = to_wgs84.transform(
            [p[0] for p in ring], [p[1] for p in ring]
        )
        return list(zip(lons, lats, strict=True))

    reader = shapefile.Reader(shp=shp_bytes, dbf=dbf_bytes)
    fields = [f[0] for f in reader.fields[1:]]
    id_idx = fields.index("LocationID") if "LocationID" in fields else 0
    zones: dict[int, dict[str, Any]] = {}
    for sr in reader.iterShapeRecords():
        loc_id = int(sr.record[id_idx])
        parts = []
        if sr.shape.parts:
            for i, start in enumerate(sr.shape.parts):
                end = (
                    sr.shape.parts[i + 1]
                    if i + 1 < len(sr.shape.parts)
                    else len(sr.shape.points)
                )
                ring = _ring(sr.shape.points[start:end])
                if ring:
                    parts.append(ring)
        else:
            parts = [_ring(sr.shape.points)]
        zones[loc_id] = {"rings": parts}
    return zones


def build_zone_table(
    lookup_csv: Path, zones_zip: Path
) -> tuple[pd.DataFrame, dict[int, ZoneInfo], dict[str, Any]]:
    """Join lookup + shapefile into per-zone metadata and a GeoJSON dict."""
    lookup = load_zone_lookup(lookup_csv)
    shapes = load_zone_shapes(zones_zip)

    rows: list[dict[str, Any]] = []
    infos: dict[int, ZoneInfo] = {}
    features = []
    for loc_id, row in lookup.iterrows():
        loc_id = int(loc_id)
        shape = shapes.get(loc_id, {"rings": []})
        rings = shape["rings"]
        lat, lon = _polygon_centroid(rings) if rings else (float("nan"),) * 2
        borough = (
            str(row["borough"])
            if pd.notna(row["borough"])
            else "Unknown"
        )
        zone_name = (
            str(row["zone"]) if pd.notna(row["zone"]) else "Unknown"
        )
        infos[loc_id] = ZoneInfo(loc_id, zone_name, borough, lat, lon)
        rows.append(
            {
                "location_id": loc_id,
                "zone": zone_name,
                "borough": borough,
                "lat": lat,
                "lon": lon,
            }
        )
        geom = {
            "type": "Polygon",
            "coordinates": [
                [[x, y] for x, y in ring] for ring in rings[:1]
            ]
            + [
                [[x, y] for x, y in ring] for ring in rings[1:]
            ],
        } if rings else {"type": "Polygon", "coordinates": []}
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "location_id": loc_id,
                    "zone": zone_name,
                    "borough": borough,
                },
                "geometry": geom,
            }
        )

    table = pd.DataFrame(rows).set_index("location_id").sort_index()
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    return table, infos, geojson
