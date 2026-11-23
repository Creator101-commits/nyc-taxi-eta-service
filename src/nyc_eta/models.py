"""Models: zone-pair median baseline and LightGBM on log1p(minutes).

The grid is frozen in config.MODEL_GRID; each config is evaluated on the
validation months by minutes-scale MAE with early stopping, and every
config's best_iteration plus a cap_bound flag is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from nyc_eta import config

LOG_EPS = 1e-6


def log1p_target(minutes: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(minutes, dtype=np.float64))


def expm1_pred(log_pred: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(log_pred, dtype=np.float64))


def mae_minutes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


class ZonePairBaseline:
    """Median minutes by (pu, do) zone pair with borough and global fallbacks."""

    def __init__(self) -> None:
        self.pair_medians: dict[tuple[int, int], float] = {}
        self.borough_medians: dict[tuple[str, str], float] = {}
        self.global_median = 10.0

    def fit(self, df: pd.DataFrame) -> ZonePairBaseline:
        g = df.groupby(["pu_zone", "do_zone"])["minutes"].median()
        self.pair_medians = {
            (int(p), int(d)): float(v) for (p, d), v in g.items()
        }
        b = df.groupby(["pu_borough", "do_borough"])["minutes"].median()
        self.borough_medians = {
            (str(p), str(d)): float(v) for (p, d), v in b.items()
        }
        self.global_median = float(df["minutes"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=np.float64)
        for i, row in enumerate(df.itertuples()):
            key = (int(row.pu_zone), int(row.do_zone))
            if key in self.pair_medians:
                out[i] = self.pair_medians[key]
            else:
                bkey = (str(row.pu_borough), str(row.do_borough))
                out[i] = self.borough_medians.get(bkey, self.global_median)
        return out


@dataclass
class LGBMConfig:
    num_leaves: int
    min_data_in_leaf: int


@dataclass
class TrainedModel:
    booster: lgb.Booster
    config: LGBMConfig
    best_iteration: int
    cap_bound: bool
    val_mae: float


def _feval_minutes_mae(
    preds: np.ndarray, dataset: lgb.Dataset
) -> tuple[str, float, bool]:
    """Evaluation metric: MAE in minutes (preds/labels are log1p space).

    LightGBM feval returns (name, value, maximize).
    """
    y_log = np.asarray(dataset.get_label(), dtype=np.float64)
    value = mae_minutes(np.expm1(y_log), np.expm1(preds))
    return ("mae_minutes", float(value), False)


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: LGBMConfig,
    seed: int = 0,
) -> TrainedModel:
    """Train one grid config with early stopping on validation minutes-MAE."""
    cat_cols = [c for c in config.CATEGORICAL_FEATURES]
    feats = config.FEATURE_COLUMNS
    X_tr = train_df[feats]
    X_va = val_df[feats]
    y_tr = log1p_target(train_df["minutes"])
    y_va_log = log1p_target(val_df["minutes"])

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_va, label=y_va_log, categorical_feature=cat_cols,
                       reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": config.LEARNING_RATE,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "num_threads": 4,
        "verbosity": -1,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=config.MAX_ROUNDS,
        valid_sets=[dval],
        feval=_feval_minutes_mae,
        callbacks=[
            lgb.early_stopping(config.EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = int(booster.best_iteration or booster.num_trees())
    # cap_bound: early stopping never fired and the round cap was reached.
    cap_bound = (
        booster.best_iteration is None
        and booster.num_trees() >= config.MAX_ROUNDS
    )
    pred_min = expm1_pred(np.asarray(booster.predict(X_va, num_iteration=best_iter)))
    return TrainedModel(
        booster=booster,
        config=cfg,
        best_iteration=best_iter,
        cap_bound=cap_bound,
        val_mae=mae_minutes(np.expm1(y_va_log), pred_min),
    )


def train_lgbm_fixed(
    train_df: pd.DataFrame, cfg: LGBMConfig, iterations: int, seed: int = 0
) -> TrainedModel:
    """Retrain for exactly ``iterations`` rounds (frozen count, no early stop)."""
    cat_cols = [c for c in config.CATEGORICAL_FEATURES]
    feats = config.FEATURE_COLUMNS
    dtrain = lgb.Dataset(
        train_df[feats], label=log1p_target(train_df["minutes"]),
        categorical_feature=cat_cols,
    )
    params = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": config.LEARNING_RATE,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "num_threads": 4,
        "verbosity": -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=iterations)
    return TrainedModel(
        booster=booster, config=cfg, best_iteration=iterations,
        cap_bound=False, val_mae=float("nan"),
    )


def select_model(
    train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int = 0
) -> tuple[TrainedModel, list[dict[str, float]]]:
    """Run the frozen grid; return the best config and per-config records."""
    records: list[dict[str, float]] = []
    best: TrainedModel | None = None
    for cfg_dict in config.MODEL_GRID:
        cfg = LGBMConfig(**cfg_dict)
        model = train_lgbm(train_df, val_df, cfg, seed=seed)
        records.append(
            {
                "num_leaves": cfg.num_leaves,
                "min_data_in_leaf": cfg.min_data_in_leaf,
                "best_iteration": model.best_iteration,
                "cap_bound": model.cap_bound,
                "val_mae": model.val_mae,
            }
        )
        if best is None or model.val_mae < best.val_mae:
            best = model
    assert best is not None
    return best, records


def predict_minutes(
    model: TrainedModel | ZonePairBaseline, df: pd.DataFrame
) -> np.ndarray:
    if isinstance(model, ZonePairBaseline):
        return model.predict(df)
    log_pred = model.booster.predict(
        df[config.FEATURE_COLUMNS], num_iteration=model.best_iteration
    )
    return expm1_pred(np.asarray(log_pred))


def save_booster(model: TrainedModel, path: str) -> None:
    model.booster.save_model(path, num_iteration=model.best_iteration)


def load_booster(path: str, best_iteration: int) -> lgb.Booster:
    booster = lgb.Booster(model_file=path)
    return booster
