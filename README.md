# NYC Taxi Trip-Time Service

A reproducible study of NYC yellow-taxi trip times: a LightGBM ETA model,
label-free drift alarms, retraining-policy simulation, a FastAPI service
and a dark single-page map website.

The question: **do label-free drift alarms on a deployed trip-time model
predict the months when its error rises, and does drift-triggered
retraining beat a static model and a fixed monthly schedule?**

## Status

- Package, tests, CI, web app, Dockerfile and Kaggle kernels implemented.
- Smoke kernel run in progress; the full benchmark has not been launched.
- See `PLAN.md` for the task list and `REVIEW.md` for verification notes.

## Protocol (`nyc-taxi-eta-v1`, frozen in `src/nyc_eta/config.py`)

- **Period:** 2019-01 through 2021-12, including the COVID shock. The
  window was chosen with hindsight to include COVID; this is disclosed
  here and in PLAN.md.
- **Splits:** train 2019-01..04, validation 2019-05..06, deployment
  simulation 2019-07..2021-12 (30 months). A final served model trains
  on 2021-10..2021-12.
- **Data:** TLC yellow-taxi monthly Parquet files from the public
  cloudfront bucket, plus `taxi_zone_lookup.csv` and `taxi_zones.zip`.
  Every downloaded file records URL, sha256, size and Last-Modified in
  the run manifest.
- **Cleaning (identical every month, counts after each filter recorded):**
  1. pickup timestamp inside the file's month;
  2. duration in [1, 180] minutes;
  3. pickup and dropoff zone IDs in 1..263 (264/265 dropped);
  4. dropoff not before pickup.
- **Sampling:** uniform per month with a fixed per-month seed; 200,000
  cleaned trips per month (smoke: 20,000).
- **Features (request-time only):** pickup/dropoff zone and borough,
  same-zone flag, airport flags (JFK 132, LGA 138, EWR 1), haversine
  distance, road distance between zone centroids from OSMnx shortest
  paths on the NYC drive graph (263x263 directed pairs), pickup hour,
  day of week, US federal holiday flag. Post-trip fields (trip_distance,
  fares, tips, dropoff time, payment type, ...) are in
  `FORBIDDEN_FEATURES` and rejected by `assert_no_leakage`.
- **Baseline:** median minutes by zone pair, falling back to borough
  pair, then global median.
- **Main model:** LightGBM predicting `log1p(minutes)`. Frozen grid:
  `num_leaves` in {63, 255} x `min_data_in_leaf` in {50, 500}, learning
  rate 0.05, early stopping 100 rounds on validation minutes-MAE, max
  3000 rounds. Selection uses validation only; every config records
  `best_iteration` and `cap_bound`.
- **Conformal:** split-conformal 80% interval on the log1p scale from
  validation residuals.
- **Drift signals (label-free):** Evidently `DataDriftPreset`
  drifted-feature share, prediction-distribution Wasserstein distance,
  trip-volume ratio. Each signal's alarm threshold is the maximum over
  the two validation months (pre-specified). A month is degraded when its
  MAE exceeds validation MAE by more than 10%.
- **Policies:** static model; monthly retrain on the trailing three
  labeled months; drift-triggered retrain on the same window only when
  the previous month's signals alarmed. Labels arrive one month late for
  every policy.
- **Metrics:** MAE, RMSE, median APE, borough slices; day-cluster
  bootstrap CI over monthly MAE; Spearman correlation between each
  signal and the MAE ratio over the 30 deployment months; alarm
  precision/recall for degraded months; SHAP TreeExplainer global
  importance on a fixed sample, pre-COVID vs April 2020.
- **Leakage/order controls:** `analysis_plan.json` is persisted before
  the first deployment-month file is read; `manifest.json` records
  `ordering_ok` (`card_write_epoch < first_deploy_read_epoch`) and the
  exact source git SHA.
- **MLflow:** every train or retrain is an MLflow run (parameters,
  metrics, model) tracked in a SQLite backend inside the kernel output.
  Browse it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## OSM routing data

Road distances and website routes come from a one-off OSMnx download of
the NYC `drive` graph (OSMnx 2.1.1), saved as GraphML in the private
Kaggle dataset `sreeharshak101/nyc-taxi-eta-osm-graph` together with the
263x263 distance table (`distances.npz`), the graph sha256 and the
Overpass `osm_base` timestamp. OpenStreetMap data is used under ODbL;
see https://www.openstreetmap.org/copyright.

## Website

`web/` is a Vite + TypeScript single page: a dark MapLibre map with taxi
zone polygons, animated taxis moving along **illustrative** routes (the
OSM drive-graph shortest paths between zone centroids; TLC data has
zones and timestamps, not GPS traces), a deployment-drift chart and a
live prediction card backed by `/predict` when the API is running.

The served model is trained on 2021-10..2021-12, so predictions reflect
late-2021 traffic patterns.

## Service and Docker

```bash
# Fixture model for local checks (no data downloads).
uv run python scripts/make_fixture_bundle.py model_bundle
uv run nyc-eta serve --host 0.0.0.0 --port 8000
```

- `GET /health`, `GET /ready` -> service status and model version.
- `GET /zones` -> all 265 zone records (id, name, borough, lat, lon).
- `GET /model-info` -> training window, validation metrics, git sha.
- `POST /predict` with `{pickup_zone_id, dropoff_zone_id,
  pickup_datetime}` -> `{minutes, interval_80: {low, high},
  model_version, route}`. Zone IDs outside 1..263 (422), unknown zones
  and malformed datetimes are rejected; post-trip fields cannot be
  supplied. `route` is reserved and currently null: the service serves
  zone-centroid distances, not turn-by-turn geometry.
- The image serves the API and the built site at `/`:

```bash
cd web && npm ci && npm run build
cd .. && uv run python scripts/make_fixture_bundle.py model_bundle
docker build -t nyc-eta-service .
docker run -p 8000:8000 nyc-eta-service
```

Deployment target is undecided (a Hugging Face Docker Space or another
host); the Dockerfile is delivery-ready.

## Kaggle runs

All data processing, training and drift computation runs in private CPU
Kaggle kernels; nothing heavy runs on the laptop.

- `scripts/build_graph_dataset.sh` (one-off): builds the OSM graph
  dataset via `kaggle_graph/`.
- `scripts/kaggle_run.sh --smoke`: pushes `kaggle_smoke/` (3 months, one
  per split, 20k samples/month).
- `scripts/kaggle_run.sh`: pushes `kaggle/` (full 36-month protocol).
- Both scripts archive `HEAD` into the private source dataset
  `sreeharshak101/nyc-taxi-eta-src`, wait for propagation, push the
  kernel, poll to completion, download output to `artifacts/kaggle/<ts>/`
  and verify `manifest.json.git_sha` equals `HEAD`.

Smoke-mode scores come from tiny samples and exercise the pipeline and
gates only; they do not drive any configuration choice. The full run is
launched only after the smoke report is reviewed.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src

cd web && npm ci && npm run typecheck && npm test && npm run build
```

## Layout

- `src/nyc_eta/` - package: `data` (TLC downloads/cleaning/sampling),
  `zones` (lookup + shapefile + GeoJSON), `features`, `routing` (OSMnx),
  `models` (baseline + LightGBM), `conformal`, `metrics`, `drift`,
  `policies`, `tracking` (MLflow), `service` (FastAPI), `experiments`
  (protocol orchestration), `cli`.
- `kaggle/`, `kaggle_smoke/`, `kaggle_graph/` - kernel entrypoints and
  metadata.
- `scripts/` - `kaggle_run.sh`, `build_graph_dataset.sh`,
  `make_fixture_bundle.py`.
- `web/` - Vite + TypeScript site; `web/public/data/` holds generated
  files copied unedited from kernel output (fixtures until then).
- `results/` - unedited kernel output (populated after runs).
- `tests/` - fixture-only unit tests; no downloads.

## License

MIT (see `LICENSE`). TLC data is published by the NYC Taxi and Limousine
Commission; OSM data is ODbL.
