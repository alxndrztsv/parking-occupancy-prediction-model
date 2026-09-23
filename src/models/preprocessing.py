from typing import Final

import pandas as pd
import polars as pl

NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "terminal_code",
        "timestamp",
        "date",
        "is_training_origin",
        "is_weekend",
        "is_bank_holiday",
        "is_closed_day",
    }
)


def get_feature_columns(schema: pl.Schema) -> list[str]:
    """Return numeric and boolean feature column names for modeling."""
    feature_columns = []

    for name, dtype in schema.items():
        if name in NON_FEATURE_COLUMNS:
            continue

        if name.startswith(("y_", "pred_")):
            continue

        if dtype.is_numeric() or dtype == pl.Boolean:
            feature_columns.append(name)

    return feature_columns


def encode_terminal_codes(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Encode terminal_code into integer terminal_id using training terminals only."""
    terminal_map = (
        train.select("terminal_code")
        .unique()
        .sort("terminal_code")
        .with_row_index("terminal_id")
        .with_columns(pl.col("terminal_id").cast(pl.Int64))
    )

    train_enc = train.join(terminal_map, on="terminal_code", how="left")
    valid_enc = (
        valid.join(terminal_map, on="terminal_code", how="left")
        .filter(pl.col("terminal_id").is_not_null())
    )
    test_enc = (
        test.join(terminal_map, on="terminal_code", how="left")
        .filter(pl.col("terminal_id").is_not_null())
    )

    return train_enc, valid_enc, test_enc, terminal_map


def prepare_pandas_split(
    df: pl.DataFrame,
    feature_columns: list[str],
    target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert Polars split into pandas format."""
    df_pd = df.select(feature_columns + [target]).to_pandas()

    X = df_pd[feature_columns]
    y = df_pd[target]

    return X, y