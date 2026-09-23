import polars as pl

from src import config


def validate_normalized(df: pl.DataFrame) -> pl.DataFrame:
    """Validate normalized ingested dataset."""
    if df.height == 0:
        raise ValueError("Normalized dataset is empty")

    critical_present = [col for col in config.CRITICAL_COLUMNS if col in df.columns]
    null_summary = df.select([pl.col(c).is_null().sum().alias(c) for c in critical_present]).to_dicts()[0]
    
    nulls = {col: cnt for col, cnt in null_summary.items() if cnt > 0}
    if nulls:
        raise ValueError(f"Null values found in normalized critical columns: {nulls}")

    return df


def validate_enriched(df: pl.DataFrame) -> pl.DataFrame:
    """Validate enriched transaction table schema, key nulls, and logical bounds."""
    if df.height == 0:
        raise ValueError("Enriched dataset is empty")

    # Check all required columns exist in schema
    missing_columns = [
        column
        for column in config.ENRICHED_REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Critical non-null columns that break time-series construction if missing
    critical_present = [col for col in config.CRITICAL_COLUMNS if col in df.columns]

    null_summary = df.select(
        [pl.col(c).is_null().sum().alias(c) for c in critical_present]
    ).to_dicts()[0]

    nulls = {col: cnt for col, cnt in null_summary.items() if cnt > 0}
    if nulls:
        raise ValueError(f"Null values found in critical columns: {nulls}")

    # Domain value constraints
    if "amount" in df.columns:
        negative_amounts = df.filter(pl.col("amount") < 0).height
        if negative_amounts > 0:
            raise ValueError(f"Found {negative_amounts} rows with negative amounts")

    if "total_spaces" in df.columns:
        invalid_spaces = df.filter(pl.col("total_spaces") <= 0).height
        if invalid_spaces > 0:
            raise ValueError(f"Found {invalid_spaces} rows with invalid total_spaces (<= 0)")

    return df