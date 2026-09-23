from pathlib import Path

import polars as pl

from src import config


def get_normalized_files(normalized_dir: Path = config.NORMALIZED_DIR) -> list[Path]:
    """Recursively fetch raw files from NORMALIZED_DIR and its subdirectories."""
    files = sorted(normalized_dir.rglob(config.RAW_GLOB))

    if not files:
        raise FileNotFoundError(f"No raw files found in {normalized_dir}")

    return files


def scan_normalized_files(normalized_dir: Path = config.NORMALIZED_DIR) -> pl.DataFrame:
    """Lazily scan raw CSV files."""
    paths = get_normalized_files(normalized_dir)
    
    return pl.scan_csv(
        paths,
        ignore_errors=True,
    )