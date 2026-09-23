import polars as pl

from src import config


def load_reference_data() -> pl.DataFrame:
    """Load, standardise and join reference tables."""
    
    coords_df = pl.read_csv(config.COORDS_PATH).rename(config.COLUMN_RENAME_MAP_COORDS)
    spaces_df = pl.read_csv(config.SPACES_PATH).rename(config.COLUMN_RENAME_MAP_SPACES)

    if "terminal_code" not in coords_df.columns or "terminal_code" not in spaces_df.columns:
        raise ValueError("Reference files must contain 'terminal_code' column.")

    # Filter non-null fields
    coords_df = coords_df.filter(
        pl.col("terminal_code").is_not_null() &
        pl.col("latitude").is_not_null() &
        pl.col("longitude").is_not_null()
    )

    spaces_df = spaces_df.filter(
        pl.col("terminal_code").is_not_null() &
        pl.col("total_spaces").is_not_null()
    )
        
    coords_df = coords_df.with_columns(pl.col("terminal_code").cast(pl.Int32))
    spaces_df = spaces_df.with_columns(pl.col("terminal_code").cast(pl.Int32))
    
    # Inner join reference tables first
    return coords_df.join(spaces_df, on="terminal_code", how="inner")


def enrich(clean_df: pl.DataFrame) -> pl.DataFrame:
    """"Join reference data with clean dataframe."""

    ref_df = load_reference_data()

    enriched_df = clean_df.join(ref_df, on="terminal_code", how="inner")

    enriched_df = enriched_df.with_columns(
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
        pl.col("total_spaces").cast(pl.Int32, strict=False),
    )

    # Check required columns
    missing_columns = [
        col for col in config.ENRICHED_REQUIRED_COLUMNS if col not in enriched_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required enriched columns: {missing_columns}")

    return enriched_df