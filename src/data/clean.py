import polars as pl

from src import config


def downcast_numeric(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast 64-bit numbers to 32-bit to halve memory consumption."""
    return df.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize, filter, parse types, and deduplicate transaction logs."""

    # Trim whitespace and artifacts from normalized column headers
    df = df.rename({col: col.strip() for col in df.columns})

    # Drop empty columns.
    empty_cols = [
        col for col in df.columns
        if df.get_column(col).null_count() == df.height
    ]
    
    if empty_cols:
        df = df.drop(empty_cols)

    # Check required columns
    missing_columns = [
        col for col in config.NORMALIZED_REQUIRED_COLUMNS if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Filter essential non-null fields
    df = df.filter(
        pl.col("terminal_code").is_not_null() &
        pl.col("start_date").is_not_null()
    )

    # Type casting
    cast_exprs = [
        pl.col("terminal_code").cast(pl.Int32, strict=False),
        pl.col("park_code").cast(pl.Int32, strict=False),
        pl.col("amount").cast(pl.Float64, strict=False),
        pl.col("paid_duration_mins").cast(pl.Int32, strict=False),
        pl.col("start_date").str.to_datetime(config.NORMALIZED_DATE_FORMAT, strict=False),
    ]

    # Clean string columns
    for col in ("zone_desc", "circuit_desc", "address", "payment_mean"):
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Utf8).str.strip_chars())

    df = df.with_columns(cast_exprs)

    unparsed_dates = df["start_date"].null_count()
    if unparsed_dates:
        raise ValueError(
            f"{unparsed_dates} rows have a start_date that does not match "
            f"NORMALIZED_DATE_FORMAT {config.NORMALIZED_DATE_FORMAT!r}"
        )

    # Deduplicate
    unique_subset = [
        col for col in ("terminal_code", "start_date", "system_id")
        if col in df.columns
    ]
    if unique_subset:
        df = df.unique(subset=unique_subset)

    df = downcast_numeric(df)

    return df