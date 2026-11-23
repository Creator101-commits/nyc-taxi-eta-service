# REVIEW

## Verification log

- Local (2026-11-23, Python 3.13.5 via uv): `uv run pytest` 34 passed;
  `uv run ruff check .` clean; `uv run mypy src` clean (15 files).
- Kaggle runs: pending (graph dataset build, then smoke).

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
