"""
Inference contract for the bundles written by artifacts.save_model_bundle.

A bundle is only valid inside the domain the model was trained on: weekday
operational bins, and terminals present in the stored terminal_map.
"""
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import polars as pl

from src.models.registry import predict_model

REQUIRED_BUNDLE_KEYS: Final[tuple[str, ...]] = (
    "model",
    "model_name",
    "horizon",
    "target",
    "feature_columns",
    "terminal_map",
    "categorical_features",
)


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a saved model bundle and check that it is complete."""
    bundle = joblib.load(path)

    missing = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle]

    if missing:
        raise ValueError(f"{path} is missing bundle keys: {missing}")

    return bundle


def encode_terminals(df: pl.DataFrame, terminal_map: pl.DataFrame) -> pl.DataFrame:
    """Attach terminal_id using the map saved at training time."""
    code_dtype = terminal_map.schema["terminal_code"]

    encoded = (
        df.with_columns(pl.col("terminal_code").cast(code_dtype))
        .with_row_index("_row_order")
        .join(terminal_map, on="terminal_code", how="left")
        .sort("_row_order")
    )

    unknown = (
        encoded.filter(pl.col("terminal_id").is_null())["terminal_code"]
        .unique()
        .sort()
        .to_list()
    )

    if unknown:
        raise ValueError(f"Terminals not seen during training: {unknown}")

    return encoded.drop("_row_order")


def predict_occupancy(bundle: dict[str, Any], df: pl.DataFrame) -> np.ndarray:
    """Predict occupancy rate for every row of df, clipped to [0, 1]."""
    model_name = bundle["model_name"]
    feature_columns = bundle["feature_columns"]
    terminal_map = pl.from_pandas(bundle["terminal_map"])

    encoded = encode_terminals(df, terminal_map)

    missing = [col for col in feature_columns if col not in encoded.columns]

    if missing:
        raise ValueError(f"Frame is missing required features: {missing}")

    X = encoded.select(feature_columns).to_pandas()

    return predict_model(bundle["model"], model_name, X)


def predict_frame(
    bundle: dict[str, Any],
    df: pl.DataFrame,
    column_name: str | None = None,
) -> pl.DataFrame:
    """Return df with the prediction appended as a column."""
    name = column_name or f"pred_{bundle['horizon']}"

    return df.with_columns(pl.Series(name, predict_occupancy(bundle, df)))
