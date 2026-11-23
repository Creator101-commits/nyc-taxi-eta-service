"""FastAPI prediction service.

Loads a served model bundle (booster + zone info + road distances +
conformal interval + metadata) and exposes /health and /predict, and
serves the built website at /.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nyc_eta import config
from nyc_eta.conformal import ConformalInterval, predict_interval
from nyc_eta.features import (
    FeatureBuilder,
    assert_no_leakage,
    categorical_columns,
)
from nyc_eta.models import expm1_pred, load_booster
from nyc_eta.routing import distance_dict, load_distance_table
from nyc_eta.zones import build_zone_table


class PredictRequest(BaseModel):
    pu_zone: int = Field(ge=1, le=config.N_ZONES)
    do_zone: int = Field(ge=1, le=config.N_ZONES)
    pickup_datetime: str


class PredictResponse(BaseModel):
    predicted_minutes: float
    interval_low_minutes: float
    interval_high_minutes: float
    confidence: float
    model_note: str


@dataclass
class ServedModel:
    booster: lgb.Booster
    best_iteration: int
    builder: FeatureBuilder
    interval: ConformalInterval
    metadata: dict[str, Any]


def load_served_model(bundle_dir: Path) -> ServedModel:
    """Load the served-model bundle produced by the kernel's final train."""
    meta = json.loads((bundle_dir / "model_meta.json").read_text())
    _, infos, _ = build_zone_table(
        bundle_dir / meta["zone_lookup_file"],
        bundle_dir / meta["zone_shapefile_zip"],
    )
    zone_ids, matrix = load_distance_table(bundle_dir / meta["distance_table"])
    road = distance_dict(zone_ids, matrix)
    builder = FeatureBuilder(infos, road)
    booster = load_booster(str(bundle_dir / meta["booster_file"]),
                           int(meta["best_iteration"]))
    interval = ConformalInterval(q=float(meta["conformal_q"]),
                                 alpha=float(meta.get("alpha", 0.2)))
    return ServedModel(
        booster=booster,
        best_iteration=int(meta["best_iteration"]),
        builder=builder,
        interval=interval,
        metadata=meta,
    )


def create_app(bundle_dir: Path | None = None,
               web_dist: Path | None = None) -> FastAPI:
    bundle = bundle_dir or Path(os.environ.get("NYC_ETA_BUNDLE", "model_bundle"))
    dist = web_dist or Path(os.environ.get("NYC_ETA_WEB", "web/dist"))
    served = load_served_model(bundle)

    app = FastAPI(title="NYC Taxi Trip-Time Service", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok",
                "model_trained_through": served.metadata.get("trained_through"),
                "n_zones": len(served.builder.zone_infos)}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        try:
            pickup = pd.Timestamp(req.pickup_datetime)
        except Exception as exc:
            raise HTTPException(422, f"bad pickup_datetime: {exc}") from exc
        if pickup.tzinfo is None:
            pickup = pickup.tz_localize("America/New_York")
        else:
            pickup = pickup.tz_convert("America/New_York")

        feats = served.builder.features_for_zones(
            req.pu_zone, req.do_zone, pickup
        )
        # Explicitly drop anything that could be a post-trip field.
        assert_no_leakage(list(feats.keys()))
        row = pd.DataFrame([feats])
        categorical_columns(row)
        log_pred = served.booster.predict(
            row[config.FEATURE_COLUMNS],
            num_iteration=served.best_iteration,
        )
        pred_min = float(expm1_pred(np.asarray(log_pred))[0])
        lo, hi = predict_interval(np.asarray(log_pred), served.interval)
        return PredictResponse(
            predicted_minutes=round(pred_min, 2),
            interval_low_minutes=round(float(lo[0]), 2),
            interval_high_minutes=round(float(hi[0]), 2),
            confidence=float(served.metadata.get("confidence", 0.8)),
            model_note=str(served.metadata.get(
                "model_note",
                "Predictions reflect late-2021 traffic patterns.")),
        )

    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    else:
        @app.get("/")
        def root() -> dict[str, Any]:
            return {"service": "nyc-eta", "web": "not built"}

    return app
