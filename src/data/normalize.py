from pathlib import Path

import polars as pl

from src import config


def month_from_filename(path: Path) -> tuple[int, int]:
    """Return (year, month) encoded in a file name."""
    stem = path.stem.lower()

    if "-" not in stem:
        raise ValueError(f"Cannot read a month from file name: {path.name}")

    abbreviation, _, year_suffix = stem.partition("-")
    month = config.MONTH_ABBREVIATIONS.get(abbreviation)

    if month is None or not year_suffix.isdigit():
        raise ValueError(f"Cannot read a month from file name: {path.name}")

    return 2000 + int(year_suffix), month


def normalize_file(raw_path: Path, output_path: Path) -> None:
    """Rewrite one API export into the normalized CSV schema."""
    year, month = month_from_filename(raw_path)

    # Read CSV and rename columns to standard schema
    source = pl.read_csv(raw_path, infer_schema_length=0).rename(config.COLUMN_RENAME_MAP_RAW)

    # Convert date strings to datetime objects for filtering
    source = source.with_columns(
        pl.col("start_date").str.to_datetime(config.RAW_DATE_FORMAT, strict=False),
        pl.col("end_date").str.to_datetime(config.RAW_DATE_FORMAT, strict=False),
    )

    unparsed_dates = source["start_date"].null_count()
    if unparsed_dates:
        raise ValueError(
            f"{unparsed_dates} rows have a start_date that does not match "
            f"RAW_DATE_FORMAT {config.RAW_DATE_FORMAT!r}"
        )

    # Filter to keep only records within the target month
    in_month = source.filter(
        pl.col("start_date").dt.year().eq(year) & pl.col("start_date").dt.month().eq(month)
    )

    # Process data
    normalized = (
        in_month.unique(subset=["system_id"], keep="first", maintain_order=True)
        .with_columns(
            pl.col("system_id").cast(pl.Int64, strict=False),
            pl.col("printed_id").cast(pl.Int64, strict=False),
            pl.col("terminal_code").cast(pl.Int32, strict=False),
            pl.col("start_date").dt.strftime(config.NORMALIZED_DATE_FORMAT),
            pl.col("end_date").dt.strftime(config.NORMALIZED_DATE_FORMAT),
            pl.col("paid_duration_mins").cast(pl.Int64, strict=False) // 60,
            pl.col("total_duration_mins").cast(pl.Int64, strict=False) // 60,
            pl.col("free_duration_mins").cast(pl.Int64, strict=False) // 60,
            pl.col("amount").cast(pl.Float64, strict=False),
            pl.col("currency").str.strip_chars(),
            pl.when(pl.col("payment_mean") == "COINS")
            .then(pl.lit("coins"))
            .otherwise(pl.lit("card")),
            pl.col("park_code").cast(pl.Int64, strict=False),
        ).select(config.NORMALIZED_COLUMNS)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_csv(output_path)


def normalize_raw_dir(
    raw_dir: Path = config.RAW_DIR,
    output_dir: Path = config.NORMALIZED_DIR,
) -> None:
    """
    Rewrite every API export under raw_dir into output_dir, mirroring its layout.

    A month is skipped when its normalized file is already at least as new as the
    source, so re-running after an incremental API pull only touches new months.
    """

    for raw_path in sorted(raw_dir.rglob(config.RAW_GLOB)):
        output_path = output_dir / raw_path.relative_to(raw_dir)

        if output_path.exists() and output_path.stat().st_mtime >= raw_path.stat().st_mtime:
            continue

        normalize_file(raw_path, output_path)
