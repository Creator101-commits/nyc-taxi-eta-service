"""Build a tiny fixture model bundle for local/Docker smoke tests.

Writes ``model_bundle/`` with a small LightGBM booster trained on
synthetic trips, the zone lookup + shapefile zip, the distance table and
``model_meta.json``. No network access; used by CI's docker job and for
local ``nyc-eta serve`` checks.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

from nyc_eta import models
from nyc_eta.features import FeatureBuilder
from nyc_eta.routing import save_distance_table
from nyc_eta.zones import ZoneInfo

N_ZONES = 12


def _zone_infos() -> dict[int, ZoneInfo]:
    boroughs = ["Manhattan", "Brooklyn", "Queens"]
    return {
        i: ZoneInfo(
            location_id=i,
            zone=f"Zone{i}",
            borough=boroughs[i % 3],
            lat=40.70 + 0.01 * i,
            lon=-74.00 - 0.01 * i,
        )
        for i in range(1, N_ZONES + 1)
    }


def _write_lookup_csv(path: Path) -> None:
    boroughs = ["Manhattan", "Brooklyn", "Queens"]
    lines = ["LocationID,BoroughID,Borough,Zone,service_zone"]
    for i in range(1, N_ZONES + 1):
        lines.append(f"{i},{i % 4 + 1},{boroughs[i % 3]},Zone{i},Yellow")
    path.write_text("\n".join(lines) + "\n")


def _write_zones_zip(path: Path) -> None:
    infos = _zone_infos()
    shp_path = path.with_suffix(".shp")
    w = shapefile.Writer(str(shp_path))
    w.field("LocationID", "N", size=10)
    for i, info in infos.items():
        lat, lon = info.lat, info.lon
        ring = [
            [lon - 0.005, lat - 0.005],
            [lon + 0.005, lat - 0.005],
            [lon + 0.005, lat + 0.005],
            [lon - 0.005, lat + 0.005],
            [lon - 0.005, lat - 0.005],
        ]
        w.poly([ring])
        w.record(i)
    w.close()
    with zipfile.ZipFile(path, "w") as zf:
        for ext in ("shp", "shx", "dbf"):
            p = path.with_suffix(f".{ext}")
            if p.exists():
                zf.write(p, p.name)
                p.unlink()


def _synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    infos = _zone_infos()
    pu = rng.integers(1, N_ZONES + 1, n)
    do = rng.integers(1, N_ZONES + 1, n)
    base = pd.Timestamp("2019-01-15 08:00")
    pickup = [base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 600, n)]
    dist = np.array([
        abs(infos[int(p)].lat - infos[int(d)].lat)
        + abs(infos[int(p)].lon - infos[int(d)].lon)
        for p, d in zip(pu, do, strict=True)
    ])
    minutes = np.clip(dist * 900 + rng.normal(0, 4, n), 2, 90)
    df = pd.DataFrame({
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": [
            p + pd.Timedelta(minutes=float(m))
            for p, m in zip(pickup, minutes, strict=True)
        ],
        "PULocationID": pu,
        "DOLocationID": do,
    })
    road = {(i, j): float(abs(i - j)) for i in infos for j in infos}
    builder = FeatureBuilder(infos, road)
    return builder.build_frame(df)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model_bundle")
    out.mkdir(parents=True, exist_ok=True)

    frame = _synthetic_frame()
    cfg = models.LGBMConfig(num_leaves=15, min_data_in_leaf=10)
    model = models.train_lgbm_fixed(frame.iloc[:300], cfg, iterations=10)
    models.save_booster(model, str(out / "model.txt"))

    _write_lookup_csv(out / "taxi_zone_lookup.csv")
    _write_zones_zip(out / "taxi_zones.zip")

    zone_ids = sorted(_zone_infos())
    matrix = np.full((len(zone_ids), len(zone_ids)), np.nan)
    for i, a in enumerate(zone_ids):
        for j, b in enumerate(zone_ids):
            matrix[i, j] = float(abs(a - b)) * 2.0
    save_distance_table({}, zone_ids, out / "_tmp.npz")
    np.savez_compressed(out / "distances.npz",
                        zone_ids=np.array(zone_ids), matrix=matrix)
    (out / "_tmp.npz").unlink(missing_ok=True)

    meta = {
        "booster_file": "model.txt",
        "best_iteration": 10,
        "conformal_q": 0.35,
        "alpha": 0.2,
        "zone_lookup_file": "taxi_zone_lookup.csv",
        "zone_shapefile_zip": "taxi_zones.zip",
        "distance_table": "distances.npz",
        "trained_through": "fixture",
        "model_note": "Fixture bundle for CI smoke tests; not a real model.",
        "confidence": 0.8,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"fixture bundle -> {out}")


if __name__ == "__main__":
    main()
