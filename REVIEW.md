# REVIEW

## Verification log

- Local (2026-11-23, Python 3.13.5 via uv): `uv run pytest` 39 passed;
  `uv run ruff check .` clean; `uv run mypy src` clean.
- Web (Node 24, npm 11): `npm run typecheck` clean; `npm test` 16 passed
  (3 files); `npm run build` clean, `web/dist` 1.9 MB.
- Kaggle smoke v5 at `67dd2c2` (final HEAD): `ordering_ok: true`,
  `manifest.git_sha` matches, `total_s: 52.1`, artifacts in
  `artifacts/kaggle/20261122-220647/`.

## Kaggle runs

- `kaggle_graph/` one-off kernel built the OSM drive graph and the
  263x263 directed distance table; published as private dataset
  `sreeharshak101/nyc-taxi-eta-osm-graph` (55,378 nodes, 139,262 edges,
  GraphML sha256 `9dab0452...`, distance table sha256 `d3bc842f...`).
  `osm_base` is empty: every Overpass status endpoint errored or timed
  out during the build; disclosed in README.
- Smoke v3 at `13304ce`: exposed the Evidently 0.7 drift-report parser
  bug (drift share silently NaN) and corrupt phase timings
  (`PhaseLog.mark` received durations, not start times).
- Smoke v4 at `38ace96`: timings corrected; policy metrics real after
  the `observed=True` groupby fix; `ordering_ok: true`.
- Smoke v5 at `67dd2c2` (final): adds `baseline_val_mae` (5.30 vs
  LightGBM val MAE 4.09) and uses the selected LightGBM as the static
  policy model, so static/drift policy MAE equals the deployment MAE
  (3.83) with zero retrains in the single smoke deploy month.
- Full run: not launched; awaits smoke report review.

## Smoke v5 numbers (pipeline check only, 3 months x 20k rows)

- Selected config: `num_leaves=63`, `min_data_in_leaf=50`,
  `best_iteration=339`, val MAE 4.090.
- Conformal alpha 0.2, q 0.442, validation coverage 0.800.
- Deploy month 2019-07: MAE 3.831, RMSE 6.019, median APE 0.228,
  coverage 0.814, n=20,000.
- Drift signals on 2019-07 vs validation thresholds: drift_share 0.0
  (no alarm), pred_wasserstein 0.689 vs 0.418 threshold (alarm),
  volume_ratio 1.0 (no alarm); degraded = false.
- Policies: static 3.831 (0 retrains), monthly 3.613 (1 retrain),
  drift 3.831 (0 retrains; the July alarm would retrain for August,
  which the smoke does not include).
- Bootstrap CI over monthly MAE: mean 3.820, [3.623, 3.989],
  n_boot=50 (smoke uses 50 vs 500 in the full run).
- Spearman correlations and alarm recall are NaN/undefined with one
  deploy month; SHAP and served-model training are skipped in smoke.

## Notes

- The 2019-01..2021-12 window was chosen with hindsight to include the
  COVID shock; disclosed in README/PLAN.
- Website taxi paths are illustrative routes on the OSM drive graph
  between zone centroids, not GPS observations.
- Served model trains on 2021-10..2021-12; predictions reflect
  late-2021 patterns.
- Smoke scores come from tiny samples and toy-scale runs; they verify
  the pipeline and gates, not model quality, and do not drive any
  configuration choice.
- `web/public/data/` holds synthetic fixtures until the full kernel
  overwrites them with real output; the smoke export has empty
  trips/routes because the demo days fall outside its deploy month.
