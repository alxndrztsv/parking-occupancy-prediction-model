from datetime import timedelta

import polars as pl

from src import config


def filter_valid_rows(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Remove closed hours, weekends, bank holidays, and missing targets."""
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    
    # Keep only valid operational origins
    df = df.filter(pl.col("is_training_origin"))
    
    # Drop rows where any target is missing
    target_cols = [f"y_{h}" for h in config.HORIZONS_MINUTES]
    for col in target_cols:
        df = df.filter(pl.col(col).is_not_null())
        
    return df


def split_by_date(
    df: pl.LazyFrame | pl.DataFrame,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Dynamically split into train, validation, and test based on days from the end."""
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    
    max_date = df.select(pl.col("timestamp").max().dt.date()).collect().item()
    
    # Calculate dynamic split boundaries
    test_start = max_date - timedelta(days=config.TEST_DAYS - 1)
    valid_start = test_start - timedelta(days=config.VALIDATION_DAYS)
    
    train = df.filter(pl.col("timestamp").dt.date() < valid_start)
    valid = df.filter(
        (pl.col("timestamp").dt.date() >= valid_start) & 
        (pl.col("timestamp").dt.date() < test_start)
    )
    test = df.filter(pl.col("timestamp").dt.date() >= test_start)
    
    return train, valid, test