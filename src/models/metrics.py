from typing import Final

import numpy as np
import pandas as pd
import polars as pl

NAIVE_COLUMNS: Final[tuple[str, ...]] = (
    "occupancy_same_time_1d",
    "current_occupancy_rate",
)


def evaluate_regression(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    mase_scale: float | None = None,
) -> dict[str, float]:
    """Calculate MAE, RMSE, WAPE and MASE."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    error = y_t - y_p
    abs_error = np.abs(error)

    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error**2)))

    true_total = float(np.sum(np.abs(y_t)))
    wape = float(np.sum(abs_error) / true_total) if true_total > 0 else float("nan")

    if mase_scale and mase_scale > 0:
        mase = mae / float(mase_scale)
    else:
        mase = float("nan")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "mase": round(mase, 4),
    }


def evaluate_prediction_column(
    df: pl.LazyFrame | pl.DataFrame,
    target: str,
    prediction_column: str,
    mase_scale: float | None = None,
) -> dict[str, float]:
    """Score one prediction column of a Polars frame with evaluate_regression."""
    data = df.lazy() if isinstance(df, pl.DataFrame) else df

    pair = (
        data.filter(pl.col(target).is_not_null() & pl.col(prediction_column).is_not_null())
        .select(target, prediction_column)
        .collect()
    )

    if pair.height == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "wape": float("nan"), "mase": float("nan")}

    return evaluate_regression(
        pair[target].to_numpy(),
        pair[prediction_column].to_numpy(),
        mase_scale,
    )


def naive_scale(
    df: pl.DataFrame,
    target: str,
    naive_columns: tuple[str, ...] = NAIVE_COLUMNS,
) -> float:
    """In-sample MAE of the naive forecast, used as the MASE denominator."""
    horizon_name = target.removeprefix("y_")
    candidates = (f"pred_seasonal_naive_{horizon_name}",) + tuple(naive_columns)
    naive_column = next((col for col in candidates if col in df.columns), None)

    if naive_column is None:
        return float("nan")

    scale = (
        df.filter(pl.col(target).is_not_null() & pl.col(naive_column).is_not_null())
        .select((pl.col(target) - pl.col(naive_column)).abs().mean())
        .item()
    )

    return float(scale) if scale is not None else float("nan")


def target_profile(
    df: pl.LazyFrame | pl.DataFrame,
    target: str,
    mase_scale: float | None = None,
) -> dict[str, float | int]:
    """Describe the target on one split: row count, share of zeros, mean, naive MAE."""
    data = df.lazy() if isinstance(df, pl.DataFrame) else df

    stats = (
        data.filter(pl.col(target).is_not_null())
        .select(
            n_rows=pl.len(),
            zero_share=(pl.col(target) == 0.0).mean(),
            mean_target=pl.col(target).mean(),
        )
        .collect()
    )

    naive_mae = float(mase_scale) if mase_scale else float("nan")

    if stats.height == 0 or stats["n_rows"][0] == 0:
        return {
            "n_rows": 0,
            "zero_share": float("nan"),
            "mean_target": float("nan"),
            "naive_mae": round(naive_mae, 4),
        }

    row = stats.row(0, named=True)

    return {
        "n_rows": int(row["n_rows"]),
        "zero_share": round(float(row["zero_share"]), 4),
        "mean_target": round(float(row["mean_target"]), 4),
        "naive_mae": round(naive_mae, 4),
    }


def build_comparison_table(
    model_metrics: pl.DataFrame,
    baseline_metrics: pl.DataFrame,
) -> pl.DataFrame:
    """Stack model and baseline scores into one tidy table."""
    model_rows = [
        model_metrics.select(
            kind=pl.lit("model"),
            split=pl.lit(split_name),
            model=pl.col("model"),
            horizon=pl.col("horizon"),
            mae=pl.col(f"{split_name}_mae"),
            rmse=pl.col(f"{split_name}_rmse"),
            wape=pl.col(f"{split_name}_wape"),
            mase=pl.col(f"{split_name}_mase"),
        )
        for split_name in ("valid", "test")
    ]

    return pl.concat(model_rows + [baseline_metrics]).sort(
        ["horizon", "split", "mae"],
        nulls_last=True,
    )
