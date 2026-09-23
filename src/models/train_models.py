from pathlib import Path

import polars as pl

from src import config
from src.models.artifacts import save_feature_importance, save_model_bundle
from src.models.estimators.base import BaseEstimator
from src.models.metrics import evaluate_regression, naive_scale
from src.models.preprocessing import (
    encode_terminal_codes,
    get_feature_columns,
    prepare_pandas_split,
)
from src.models.registry import CATEGORICAL_FEATURES, MODEL_NAMES, get_estimator


def _train_single_horizon(
    model_name: str,
    horizon_name: str,
    train_enc: pl.DataFrame,
    valid_enc: pl.DataFrame,
    test_enc: pl.DataFrame,
    feature_columns: list[str],
    terminal_map_pd,
    model_dir: Path | None,
    report_dir: Path | None,
    fitted_estimator: BaseEstimator | None = None,
) -> tuple[dict, BaseEstimator]:
    """Train, evaluate, and save a single model on one horizon."""
    target = f"y_{horizon_name}"
    train_h = train_enc.filter(pl.col(target).is_not_null())
    valid_h = valid_enc.filter(pl.col(target).is_not_null())
    test_h = test_enc.filter(pl.col(target).is_not_null())

    mase_scale = naive_scale(train_h, target)

    X_train, y_train = prepare_pandas_split(train_h, feature_columns, target)
    X_valid, y_valid = prepare_pandas_split(valid_h, feature_columns, target)
    X_test, y_test = prepare_pandas_split(test_h, feature_columns, target)

    if fitted_estimator is not None:
        estimator = fitted_estimator
    else:
        estimator = get_estimator(model_name)
        estimator.fit(X_train, y_train, X_valid, y_valid, list(CATEGORICAL_FEATURES))

    valid_pred = estimator.predict(X_valid)
    test_pred = estimator.predict(X_test)
    valid_metrics = evaluate_regression(y_valid, valid_pred, mase_scale)
    test_metrics = evaluate_regression(y_test, test_pred, mase_scale)

    # The estimator is saved, not the library model inside it: the estimator
    # carries the category mapping that XGBoost needs to predict correctly.
    if model_dir:
        save_model_bundle(
            model_dir=model_dir,
            model=estimator,
            model_name=model_name,
            horizon_name=horizon_name,
            target=target,
            feature_columns=feature_columns,
            terminal_map=terminal_map_pd,
        )

    if report_dir:
        save_feature_importance(
            report_dir=report_dir,
            estimator=estimator,
            model_name=model_name,
            horizon_name=horizon_name,
            feature_columns=feature_columns,
        )

    metrics_row = {
        "model": model_name,
        "horizon": horizon_name,
        "target": target,
        "valid_mae": valid_metrics["mae"],
        "valid_rmse": valid_metrics["rmse"],
        "valid_wape": valid_metrics["wape"],
        "valid_mase": valid_metrics["mase"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_wape": test_metrics["wape"],
        "test_mase": test_metrics["mase"],
    }
    return metrics_row, estimator


def train_all_models(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    test_df: pl.DataFrame,
    model_names: tuple[str, ...] = MODEL_NAMES,
    model_dir: Path | None = None,
    report_dir: Path | None = None,
) -> pl.DataFrame:
    """Orchestrate training for all models and horizons, creating ensemble blends seamlessly."""
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)

    train_enc, valid_enc, test_enc, terminal_map = encode_terminal_codes(
        train=train_df, valid=valid_df, test=test_df
    )
    feature_columns = get_feature_columns(train_enc.schema)
    terminal_map_pd = terminal_map.to_pandas()

    results = []
    has_ensemble = "ensemble" in model_names
    base_model_names = [m for m in model_names if m != "ensemble"]

    for horizon_name in config.HORIZONS_MINUTES:
        horizon_estimators: dict[str, BaseEstimator] = {}

        # Train base models for this horizon
        for model_name in base_model_names:
            row, est = _train_single_horizon(
                model_name=model_name,
                horizon_name=horizon_name,
                train_enc=train_enc,
                valid_enc=valid_enc,
                test_enc=test_enc,
                feature_columns=feature_columns,
                terminal_map_pd=terminal_map_pd,
                model_dir=model_dir,
                report_dir=report_dir,
            )
            results.append(row)
            horizon_estimators[model_name] = est

        # Blend trained models into ensemble with zero re-training overhead
        if has_ensemble and horizon_estimators:
            ensemble_est = get_estimator("ensemble")
            members = ensemble_est.params.get("models") or list(horizon_estimators)
            for m_name, m_est in horizon_estimators.items():
                if m_name in members:
                    ensemble_est.add_estimator(m_name, m_est)

            row, _ = _train_single_horizon(
                model_name="ensemble",
                horizon_name=horizon_name,
                train_enc=train_enc,
                valid_enc=valid_enc,
                test_enc=test_enc,
                feature_columns=feature_columns,
                terminal_map_pd=terminal_map_pd,
                model_dir=model_dir,
                report_dir=report_dir,
                fitted_estimator=ensemble_est,
            )
            results.append(row)

    metrics_table = pl.DataFrame(results)
    if report_dir:
        metrics_table.write_csv(report_dir / "model_metrics.csv")

    return metrics_table