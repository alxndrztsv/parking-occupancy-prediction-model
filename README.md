# Parking Occupancy Prediction

Forecasting how full municipal parking terminals will be 1, 2, 3 and 24 hours
ahead, using parking transaction logs. The goal of the project is to
compare gradient boosting models against each other and against naive baselines,
and pick the best model for each horizon.

## Problem

The raw data is a log of parking transactions: ticket issue time and paid
duration. Terminal capacity and coordinates come from separate reference tables.
From these the actual occupancy of every terminal is reconstructed on a
10-minute grid:

```
occupancy_rate = active_sessions / total_spaces
```

where `active_sessions` is the number of tickets whose paid duration covers that
10-minute bin. Parking time is only consumed inside the terminal's operating
hours (09:00–20:00); nights, weekends and bank holidays count as zero occupancy.

The model predicts `occupancy_rate` at `t + h` from the state at `t`. This is a
direct forecasting scheme: every horizon has its own model and predictions are
never fed back as inputs, so error does not accumulate over the horizon.

## Data

| Property | Value |
|---|---|
| Terminals | 308 |
| Grid interval | 10 minutes |
| Operating hours | 09:00–20:00, weekdays |
| Train | 2023-01-02 → 2026-05-29, 15,759,480 rows, 308 terminals |
| Valid | 2026-06-02 → 2026-07-29, 814,308 rows, 297 terminals |
| Test | 2026-07-30 → 2026-08-28, 387,354 rows, 283 terminals |

Not every terminal is active in every window, which is why the valid and test
splits cover fewer of them than train.

The split is strictly temporal: the last 30 days are the test set, the 60 days
before that are validation. Early stopping is tuned on validation, and the test
set is used exactly once.

Terminals are located in the Dublin / Dún Laoghaire–Rathdown area of Ireland,
which is why the bank holiday calendar in `src/config.py` is the Irish one.

Data arrives in three layers, none of which is committed:

- `data/raw/<year>/<mon-yy>.csv` — API exports exactly as delivered;
- `data/normalized/<year>/<mon-yy>.csv` — the same rows in one canonical CSV
  schema, written from `raw/` by `src/data/normalize.py`. Months that predate the
  API exist only here, so this layer is where history and the API meet;
- `data/reference/` — `coordinates.csv` and `total_spaces.csv`, joined in by
  terminal code.

## Features

24 features per row: occupancy lags (1, 2, 3, 6, 12 and 18 bins — one lag per
intraday horizon), same-time values from 1, 7 and 14 days earlier, the mean of the
same time over the last 7 days, rolling means over 1, 2 and 3 hours, rolling
standard deviation over 1 hour, calendar features, terminal capacity, and
`terminal_id` as a categorical feature.

Lags stop at 18 bins on purpose. A lag of `L` bins is structurally zero for every
row within `L` bins of the 09:00 opening, because the grid zeroes out
non-operating hours: at `L = 36` that would be 55 % of the operating rows.
Medium-term context is carried by the same-time-yesterday features instead, which
the operating-hours truncation does not affect.

## Repository layout

```
configs/models.yaml        hyperparameters for every model
src/config.py              paths, grid settings, horizons, bank holidays
src/data/                  normalize (API -> canonical CSV), ingest, clean,
                           enrich (reference join), validate
src/features/              occupancy grid, feature table, temporal split
src/models/
  estimators/              one module per algorithm (lightgbm, xgboost,
                           catboost, ensemble) behind a common BaseEstimator
  registry.py              model name -> estimator class, hyperparameter injection
  hyperparams.py           configs/models.yaml loader and validation
  preprocessing.py         feature-column selection, terminal encoding, pandas conversion
  metrics.py               MAE, RMSE, WAPE, MASE - shared by models and baselines
  baselines.py             naive forecasts
  train_models.py          training orchestration
  artifacts.py             model and feature-importance persistence
  predict.py               inference contract for saved bundles
src/pipeline.py            staged end-to-end run
app/                       FastAPI + deck.gl demo: historical day replay
tests/                     pytest suite
```

## Pipeline

`src/pipeline.py` is split into stages. Each stage writes its result to `data/`
and can be re-run independently:

| Stage | What it does | Artifact |
|---|---|---|
| `run_stage_normalize` | rewrites API exports into the canonical CSV schema; months already up to date are skipped, nothing is deleted | `data/normalized/<year>/*.csv` |
| `run_stage_loading` | reads the normalized CSVs, cleans them, joins the reference tables, validates | `data/enriched/clean.parquet` |
| `run_stage_grid` | builds the occupancy table and a continuous 24-hour grid | `data/processed/operational_table.parquet`, `data/processed/continuous_table.parquet` |
| `run_stage_features` | adds lags, rolling windows and targets | `data/processed/features.parquet` |
| `run_stage_split` | filters operating bins and splits by date | `data/splits/{train,valid,test}.parquet` |
| `run_stage_baseline` | scores naive forecasts and profiles the targets | `reports/baseline_metrics.csv`, `reports/target_profile.csv` |
| `run_stage_train` | trains every model on every horizon | `models/*.joblib`, `reports/model_metrics.csv`, `reports/*_feature_importance.csv` |
| `run_stage_compare` | merges models and baselines into one table | `reports/model_comparison.csv` |

```bash
python -m src.pipeline
```

## Models

Every model is a separate class in `src/models/estimators/` implementing a common
`BaseEstimator` interface (`fit`, `predict`, `get_feature_importances`):

- **LightGBM** — `regression_l1`; categorical `terminal_id` passed via `categorical_feature`;
- **XGBoost** — `reg:absoluteerror` with `enable_categorical`; categories are encoded
  against the list frozen at training time;
- **CatBoost** — `MAE`; categorical features passed by name;
- **Ensemble** — weighted average of already fitted models; membership is declared
  in `configs/models.yaml` (`ensemble.models`).

Hyperparameters live in `configs/models.yaml`. A shared `random_state` and
`early_stopping_rounds` are injected into every model automatically.

A saved artifact (`models/{model}_{horizon}.joblib`) contains the estimator, the
feature list and the terminal map — enough to predict through
`src/models/predict.py` without retraining.

## Results

Test split, MAE (lower is better). Persistence means "occupancy will not change".

| Horizon | LightGBM | XGBoost | CatBoost | Ensemble | Persistence | Best gain |
|---|---|---|---|---|---|---|
| 1 hour  | **0.0465** | 0.0471 | 0.0502 | 0.0472 | 0.0560 | 17 % |
| 2 hours | **0.0482** | 0.0483 | 0.0513 | 0.0484 | 0.0723 | 33 % |
| 3 hours | 0.0433 | **0.0429** | 0.0474 | 0.0434 | 0.0789 | 46 % |
| 24 hours | **0.0480** | 0.0491 | 0.0503 | 0.0486 | 0.0832 | 42 % |

LightGBM is the best model on three of the four horizons and is selected as the
final model. XGBoost wins at 3 hours by 0.9 % and is a statistical tie at 2 hours
(0.0483 vs 0.0482); both gaps are well inside the noise of a single temporal
split. The equal-weight ensemble of all three never beats the best single model on
any horizon: the boosters are
trained on the same features with the same loss, their errors are strongly
correlated, and the average simply lands between the members.

Against the horizon-specific seasonal naive, LightGBM scores MASE 0.54 / 0.62 /
0.65 / 0.50 — between 35 % and 50 % less error than the naive reference. The naive
baselines score worse on the same reference: `pred_current` 0.65–1.19,
`pred_yesterday` 0.93–1.26, `pred_last_week` 0.93–1.32, i.e. two of the three are
no better than simply taking yesterday's value.

The full per-model, per-baseline, per-split table is in
[`reports/model_comparison.csv`](reports/model_comparison.csv).

Relative accuracy (WAPE) and the share of zero targets:

| Horizon | WAPE (LightGBM) | Zero-target share |
|---|---|---|
| 1 hour | 0.51 | 0.53 |
| 2 hours | 0.59 | 0.58 |
| 3 hours | 0.61 | 0.64 |
| 24 hours | 0.64 | 0.61 |

Accuracy degrades monotonically with the horizon, which is the expected
behaviour. Horizons must not be ranked by MAE: longer horizons have a lower mean
target and a larger share of zeros, so their MAE is formally smaller even though
their relative accuracy is worse. WAPE is the comparable metric here.

## Metrics

A single implementation in `src/models/metrics.py`, shared by models and baselines:

- **MAE** — mean absolute error in occupancy shares;
- **RMSE** — penalises large misses;
- **WAPE** — total absolute error over total actual value; comparable across
  horizons; NaN when every actual value is zero;
- **MASE** — model MAE over naive-forecast MAE; below 1 means "better than naive".

The MASE denominator is horizon-specific: the seasonal naive for a target at
`t + h` is the occupancy one day before the target moment, `occupancy(t + h - 1d)`,
built as `pred_seasonal_naive_{horizon}` in the feature table. For the 24-hour
horizon it reduces to the current occupancy. MASE is therefore comparable both
between models and between horizons.

## Installation

Python 3.14, environment managed with `uv`:

```bash
uv venv
uv pip install -r requirements.txt        # runtime
uv pip install -r requirements-dev.txt    # + pytest
```

Dependency versions are pinned, and that matters here: the saved bundles are joblib
pickles and will not load under a different CatBoost or LightGBM than the one that
wrote them. By default CatBoost runs on GPU (`task_type: GPU`) while LightGBM and
XGBoost run on CPU. `cmake` and `ninja` are only needed if you build LightGBM from
source.

## Docker

The image pins the exact library versions the bundles were created with, which is
the point of running it: a bundle that fails to load on another machine usually
means that machine resolved `catboost` or `scipy` to a different version.

```bash
sudo systemctl start docker                          # if the daemon is not running
docker compose build
docker compose run --rm pipeline                     # full pipeline run
docker compose run --rm pipeline python -m pytest    # tests inside the image
```

`data/`, `models/` and `reports/` are mounted from the host rather than baked into
the image, and none of them is in the repository. A fresh clone therefore needs at
least `data/raw`, `data/normalized` and `data/reference` to run the pipeline, or
`models/` plus `data/processed/features.parquet` and `data/enriched/clean.parquet`
for inference only.

The container runs on CPU by default. Because `configs/models.yaml` asks CatBoost
for a GPU, `ML_DEVICE` overrides the device for all three libraries at once:

| `ML_DEVICE` | LightGBM `device_type` | XGBoost `device` | CatBoost `task_type` |
|---|---|---|---|
| unset | from `models.yaml` | from `models.yaml` | from `models.yaml` |
| `cpu` | `cpu` | `cpu` | `CPU` |
| `gpu` | `gpu` | `cuda` | `GPU` |

An unrecognised value fails at import with a clear message rather than silently
training on the wrong device. For GPU, install nvidia-container-toolkit, add
`gpus: all` to the service in `docker-compose.yml`, and run:

```bash
docker compose run --rm --gpus all -e ML_DEVICE=gpu pipeline
```

## Tests

```bash
python -m pytest
```

27 tests. Covered: metrics (including the invariant that a seasonal-naive baseline
scores exactly MASE = 1.0, and that the MASE denominator follows the horizon),
feature selection, terminal encoding, artifact save/load, and regressions for
categorical handling in XGBoost and CatBoost.

## Limitations

- **Zero-inflated targets.** More than half of the target values are zero, because
  the grid zeroes out nights, weekends and bank holidays, and a horizon often
  reaches past operating hours. Part of the reported accuracy is the model
  correctly predicting a closed car park. Tracked via `reports/target_profile.csv`.
- **`terminal_id` dominance.** It accounts for 36 % of LightGBM's splits:
  the model spends much of its capacity telling terminals apart rather than
  modelling occupancy dynamics. It will not transfer to an unseen terminal.
- **Single split.** Model comparison uses one temporal split. The gap between
  LightGBM and XGBoost in the third decimal of MAE is not statistically meaningful;
  a rolling-origin CV is needed for a robust ranking.