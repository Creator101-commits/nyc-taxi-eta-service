# NYC Taxi ETA Service

A reproducible NYC yellow-taxi trip-time study: LightGBM ETA model,
label-free drift alarms, retraining-policy simulation, a FastAPI service
and a dark map website with illustrative taxi paths.

## Build commands

```bash
# Verification commands.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src

# Web app.
cd web && npm ci && npm run build && npm test

# Kaggle (CPU kernels only; no laptop data processing).
scripts/build_graph_dataset.sh   # one-off OSM graph dataset
scripts/kaggle_run.sh --smoke    # smoke kernel
scripts/kaggle_run.sh            # full kernel (after smoke approval)
```

## Platform support

- macOS and Linux; Python 3.12-3.13; CPU only.
- All data downloads, cleaning, training, drift and evaluation run inside
  private Kaggle CPU kernels. Local work is fixture-based unit tests and
  the web build only.
- Docker is not installed locally; the image build and API smoke run in CI.

## Key conventions

- Frozen protocol `nyc-taxi-eta-v1`: TLC monthly parquet -> four frozen
  cleaning filters with per-filter counts -> per-month seeded uniform
  sample -> request-time features -> LightGBM grid selected on
  validation MAE only -> split-conformal 80% interval -> label-free drift
  signals -> static / monthly / drift retraining policies.
- `analysis_plan.json` is persisted before the first deployment-month
  file is read; `manifest.json` carries `ordering_ok` and the git SHA.
- Features are request-time only; `FORBIDDEN_FEATURES` (fares, tips,
  trip_distance, dropoff time, payment type, ...) are rejected by
  `assert_no_leakage`.
- The study window 2019-01..2021-12 was chosen with hindsight to include
  COVID; this is disclosed in the README.
- Website taxi paths are illustrative: TLC data has zones and timestamps,
  not GPS traces.
- The served model trains on 2021-10..2021-12 and its predictions reflect
  late-2021 traffic patterns.
- `results/` holds unedited kernel output; `image.png` renders from
  `results/` only.

## Rules

- Default branch is `main`; never force-push it or rewrite shared history.
- Ask before adding any dependency or package manager.
- Never fabricate data, benchmarks, logs or results.
- Done means the relevant tests actually ran and passed; never skip or mock them to finish.
- No emojis anywhere: code, comments, commits, READMEs, docs.
- No em dashes (the long dash, U+2014) anywhere: code, comments, commits, READMEs, docs. Use a comma, colon, parentheses, or a new sentence instead.
- Keep README in sync with code changes, following its existing template.
- Comments are brief and explain why, only when the code doesn't already say it.
- Keep dataset splits and eval leakage controls intact.
- Never run data processing, training or drift computation on the laptop;
  only tiny synthetic fixtures in tests.

## Commits

- Conventional Commits: `type(scope): summary`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`; scope is optional.
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- One logical change per commit; keep formatting-only changes separate.
- Add a body only when the reason is not obvious: explain why, not what, wrapped at 72.
- Breaking changes: `type!:` in the summary plus a `BREAKING CHANGE:` footer.
- Commit only when asked. Never commit secrets, build output, or failing code.
