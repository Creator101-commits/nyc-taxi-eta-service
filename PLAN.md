# PLAN

## Status

- [x] Package scaffold: `src/nyc_eta/` modules (data, features, routing,
      models, conformal, drift, policies, tracking, service, experiments)
- [x] Local fixture tests (39 tests), ruff, mypy
- [x] Kaggle kernels: full (`kaggle/`), smoke (`kaggle_smoke/`), one-off
      OSM graph build (`kaggle_graph/`), `scripts/kaggle_run.sh`,
      `scripts/build_graph_dataset.sh`
- [x] OSM graph dataset build (one-off kernel) -> private dataset
      `sreeharshak101/nyc-taxi-eta-osm-graph`
- [x] Smoke kernel runs: v4 verified at 38ace96; v5 at final HEAD
- [x] Web app (Vite + TypeScript + MapLibre + deck.gl), 16 vitest tests
- [x] CI workflow (python matrix, web, docker) + Dockerfile
- [ ] Full kernel run (only after smoke report is approved)
- [ ] results/ curation, image.png, README findings

## Protocol (frozen, `nyc-taxi-eta-v1`)

- Period 2019-01..2021-12 (chosen with hindsight to include COVID;
  disclosed).
- Train 2019-01..04, validation 2019-05..06, deployment 2019-07..2021-12
  (30 months). Served model trains on 2021-10..12.
- Cleaning: pickup inside file month; duration in [1,180] min; zone ids
  1..263; dropoff not before pickup. Counts after each filter recorded.
- Sample: uniform per month, fixed per-month seed, 200k rows (smoke 20k).
- Features: zones, boroughs, same-zone, airport flags, haversine +
  OSMnx drive-graph road distance, pickup hour/dow, federal holiday.
- Model: LightGBM on log1p(minutes); grid 2x2 (num_leaves 63/255,
  min_data_in_leaf 50/500), lr 0.05, early stop 100 on val minutes-MAE,
  max 3000 rounds; validation-only selection; best_iteration + cap_bound
  recorded.
- Baseline: median minutes by zone pair -> borough pair -> global.
- Conformal: split-conformal 80% interval on log1p scale from validation
  residuals.
- Drift signals (label-free): Evidently DataDriftPreset drifted share,
  prediction-distribution Wasserstein, trip-volume ratio. Alarm threshold
  per signal = max over the two validation months.
- Degraded month: MAE > 10% above validation MAE.
- Policies: static; monthly retrain on trailing 3 labeled months; drift
  retrain on the same window only when the previous month's signals
  alarmed. Labels arrive one month late for all policies.
- Metrics: MAE, RMSE, median APE, borough slices; day-cluster bootstrap
  CI over monthly MAE; Spearman(signal, MAE ratio); alarm precision /
  recall; SHAP TreeExplainer pre-COVID vs 2020-04.
- MLflow: sqlite backend in kernel output; every train/retrain is a run.

## Open questions

- Full-run runtime projection comes from the smoke kernel timings.
- Deployment target for the service is undecided (HF Docker Space or
  other host); Dockerfile is delivery-ready regardless.
