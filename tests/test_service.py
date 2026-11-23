"""Service endpoint tests against a tiny fixture model bundle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import shapefile
from fastapi.testclient import TestClient

from nyc_eta import config, models
from nyc_eta.routing import save_distance_table
from nyc_eta.service import create_app


def _write_zones_zip(path: Path, n: int = 10, strict=False) -> None:
    """Minimal valid shapefile zip: one square polygon per LocationID."""
    shp_path = path.with_suffix(".shp")
    w = shapefile.Writer(str(shp_path))
    w.field("LocationID", "N", size=10)
    for i in range(1, n + 1):
        lat = 40.70 + 0.01 * i
        lon = -74.00 - 0.01 * i
        # ~0.005-degree square around the centroid
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


def _write_lookup_csv(path: Path, n: int = 10) -> None:
    boroughs = ["Manhattan", "Brooklyn", "Queens"]
    lines = ["LocationID,BoroughID,Borough,Zone,service_zone"]
    for i in range(1, n + 1):
        lines.append(
            f"{i},{i % 4 + 1},{boroughs[i % 3]},Zone{i},Yellow"
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def bundle(tmp_path: Path, feature_frame) -> Path:
    """A real (tiny) served-model bundle in tmp_path."""
    b = tmp_path / "bundle"
    b.mkdir()

    train = feature_frame.iloc[:150]
    cfg = models.LGBMConfig(num_leaves=15, min_data_in_leaf=10)
    model = models.train_lgbm_fixed(train, cfg, iterations=5)
    models.save_booster(model, str(b / "model.txt"))

    _write_lookup_csv(b / "taxi_zone_lookup.csv")
    _write_zones_zip(b / "taxi_zones.zip", strict=False)

    zone_ids = list(range(1, 11))
    mat = np.full((10, 10), np.nan)
    for i in range(10):
        for j in range(10):
            mat[i, j] = 0.5 + 0.1 * abs(i - j)
    save_distance_table({}, zone_ids, b / "_tmp.npz")  # placeholder shape
    # rewrite with real matrix via npz directly
    np.savez_compressed(
        b / "distances.npz",
        zone_ids=np.array(zone_ids),
        matrix=mat,
    )
    (b / "_tmp.npz").unlink(missing_ok=True)

    meta = {
        "booster_file": "model.txt",
        "best_iteration": 5,
        "conformal_q": 0.4,
        "alpha": 0.2,
        "zone_lookup_file": "taxi_zone_lookup.csv",
        "zone_shapefile_zip": "taxi_zones.zip",
        "distance_table": "distances.npz",
        "trained_through": "2021-12",
        "model_note": "fixture bundle",
        "confidence": 0.8,
    }
    (b / "model_meta.json").write_text(json.dumps(meta))
    return b


@pytest.fixture
def client(bundle, tmp_path) -> TestClient:
    web = tmp_path / "webdist"
    web.mkdir()
    (web / "index.html").write_text("<html>fixture</html>")
    app = create_app(bundle_dir=bundle, web_dist=web)
    return TestClient(app)


class TestService:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_predict_ok(self, client):
        r = client.post("/predict", json={
            "pu_zone": 1, "do_zone": 2,
            "pickup_datetime": "2019-07-04T08:30:00",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["predicted_minutes"] > 0
        assert body["interval_low_minutes"] <= body["predicted_minutes"]
        assert body["interval_high_minutes"] >= body["predicted_minutes"]
        assert "late-2021" in body["model_note"] or "fixture" in body["model_note"]

    def test_predict_rejects_out_of_range_zone(self, client):
        r = client.post("/predict", json={
            "pu_zone": 999, "do_zone": 2,
            "pickup_datetime": "2019-07-04T08:30:00",
        })
        assert r.status_code == 422

    def test_predict_rejects_bad_datetime(self, client):
        r = client.post("/predict", json={
            "pu_zone": 1, "do_zone": 2,
            "pickup_datetime": "not-a-date",
        })
        assert r.status_code == 422

    def test_web_index_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "fixture" in r.text

    def test_no_forbidden_fields_in_request_model(self):
        # the request schema must not accept post-trip fields
        from nyc_eta.service import PredictRequest

        fields = set(PredictRequest.model_fields)
        assert fields == {"pu_zone", "do_zone", "pickup_datetime"}
        assert not (fields & config.FORBIDDEN_FEATURES)
