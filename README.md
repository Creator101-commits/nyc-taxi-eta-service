# NYC Taxi Trip-Time Service

> Predicting NYC taxi trip times across 36 months including COVID: retraining when drift alarms fire cut the error by 0.41 minutes per trip.

Do label-free drift alarms on a deployed trip-time model predict the
months when its error rises, and does drift-triggered retraining beat a
static model and a fixed monthly schedule? This project answers that on
36 months of NYC yellow-taxi data (2019-01 to 2021-12, including the
COVID shock), simulating deployment over the final 30 months
(2019-07 onward), with a LightGBM ETA model, a retraining-policy
simulation, a FastAPI service and a dark single-page map website.

![NYC taxi ETA results: monthly MAE by retraining policy through the COVID collapse, paired policy bootstrap differences, drift-signal lead/lag correlations and interval coverage](image.png)

The figure follows the deployment simulation: (a) monthly MAE by
retraining policy through the COVID collapse, (b) paired policy
bootstrap differences, (c) drift-signal lead/lag correlations, (d) the
static model's 80% interval coverage.

## Results

Mean monthly MAE over the 30 deployment months, paired day-cluster
bootstrap 95% CI (1,000 replicates, seed 0):

| policy | mean MAE | 95% CI | retrains |
| :--- | ---: | ---: | ---: |
| static | 3.701 | [3.675, 3.730] | 0 |
| monthly | 3.262 | [3.238, 3.287] | 30 |
| drift | 3.295 | [3.269, 3.321] | 22 |

- **Retraining is what survives the shock.** Monthly beats static by
  -0.440 [-0.456, -0.423] and drift-triggered by -0.406 [-0.422,
  -0.390] minutes per trip; all three paired CIs exclude zero.
- **Drift gets most of the gain with fewer retrains, but not all.** 22
  retrains recover about 92% of the monthly gain; monthly still wins by
  +0.033 [+0.028, +0.039].
- **The alarms detect but cannot predict.** Any-signal alarms fired in
  23 of 30 months: recall 1.0 on the four degraded months but precision
  0.17 (19 false alarms), and stayed on for 20 straight months after
  the error recovered. Lead/lag Spearman correlations are
  contemporaneous: no signal leads the error by a month.
- **In the COVID collapse a simple zone-pair median beat the static
  model** in four months (2020-04 to 2020-06, 2021-01), and static 80%
  interval coverage fell to 0.52 before recovering.

| drift signal | lag -1 | lag 0 | lag +1 |
| :--- | ---: | ---: | ---: |
| drift_share | 0.376 | 0.460 | 0.416 |
| pred_wasserstein | 0.173 | 0.240 | 0.175 |
| volume_ratio | -0.307 | -0.404 | -0.353 |

## How it works

```text
TLC monthly parquet -> four cleaning filters -> 200,000 seeded
rows/month -> request-time features only (post-trip fields rejected)
-> LightGBM grid selected on validation MAE -> split-conformal 80%
interval -> label-free drift signals -> static / monthly / drift
retraining policies
```

`analysis_plan.json` is persisted before the first deployment-month
read, and `manifest.json` records `ordering_ok` plus the git SHA.

## Run it

```bash
uv sync
cd web && npm ci && npm run build && cd ..
uv run nyc-eta serve --host 0.0.0.0 --port 8000
```

```bash
uv run pytest
uv run ruff check .
uv run mypy src
cd web && npm ci && npm run typecheck && npm test && npm run build
```

The full benchmark runs on Kaggle CPU (the laptop only builds, tests
and lints):

```bash
scripts/kaggle_run.sh --smoke   # pipeline check first
scripts/kaggle_run.sh           # full 36-month protocol
```

## Use it

Live site (static taxi replay and drift chart):
https://creator101-commits.github.io/nyc-taxi-eta-service/

The served model is trained on 2021-10..2021-12, so predictions reflect
late-2021 traffic patterns. With `nyc-eta serve` running:

```bash
curl -s localhost:8000/zones | head
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "pickup_zone_id": 132, "dropoff_zone_id": 138,
  "pickup_datetime": "2021-12-15T08:30:00"}'
```

## Notes

- The animated taxi routes are illustrative: TLC data has zones and
  timestamps, not GPS traces. The live predictor needs the API, which
  static GitHub Pages hosting does not run.
- The pre-registered trip-volume alarm fires only on increases, so it
  never fired during the COVID collapse (volume ratio about 0.03); a
  post-hoc two-sided rule is reported separately.
- The study window 2019-01..2021-12 was chosen with hindsight to
  include COVID, and alarm thresholds are the max over only two
  validation months.
- Data: NYC TLC yellow-taxi monthly Parquet plus taxi zone files;
  protocol `nyc-taxi-eta-v1`, bootstrap 1,000 replicates seed 0.
  Primary run v1.1 (`e18c996`); `results/` is unedited kernel output
  and `image.png` renders from it.
