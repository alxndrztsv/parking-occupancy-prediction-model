from typing import Final

import polars as pl

from src.models import metrics

BASELINE_MODELS: Final[tuple[str, ...]] = (
    "pred_current",
    "pred_yesterday",
    "pred_last_week",
)

def evaluate_baselines(
    df: pl.LazyFrame | pl.DataFrame,
    target: str,
    prediction_columns: tuple[str, ...] = BASELINE_MODELS,
    mase_scale: float | None = None,
) -> pl.DataFrame:
    """Evaluate baseline predictions with the shared regression metrics."""
    data = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Add baseline columns
    data = data.with_columns([
        pl.col("current_occupancy_rate").alias("pred_current"),
        pl.col("occupancy_same_time_1d").alias("pred_yesterday"),
        pl.col("occupancy_same_time_7d").alias("pred_last_week"),
    ]).collect()

    rows = [
        {
            "model": pred,
            **metrics.evaluate_prediction_column(data, target, pred, mase_scale),
        }
        for pred in prediction_columns
    ]

    return pl.DataFrame(rows)


def evaluate_baselines_by_split(
    splits: dict[str, pl.DataFrame],
    target: str,
    horizon_name: str,
    mase_scale: float | None = None,
) -> pl.DataFrame:
    """Score every baseline on every split in tidy long format.

    The MASE denominator is computed once on the training split and passed in, so
    baseline and model scores are directly comparable.
    """
    frames = [
        evaluate_baselines(df, target=target, mase_scale=mase_scale).with_columns(
            kind=pl.lit("baseline"),
            split=pl.lit(split_name),
            horizon=pl.lit(horizon_name),
        )
        for split_name, df in splits.items()
    ]

    return pl.concat(frames).select(
        "kind", "split", "model", "horizon", "mae", "rmse", "wape", "mase"
    )