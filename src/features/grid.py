import polars as pl
import polars.selectors as cs

from src import config


def downcast_numeric[T: (pl.DataFrame, pl.LazyFrame)](df: T) -> T:
    """Downcast 64-bit float and integer columns to 32-bit to reduce memory footprint."""
    return df.with_columns(
        cs.by_dtype(pl.Float64).cast(pl.Float32),
        cs.by_dtype(pl.Int64).cast(pl.Int32),
    )


def build_operational_occupancy_table(
    data: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """
    Build a 10-minute terminal occupancy table using terminal
    operational hours only: 09:00 to 20:00.

    Rules:
        - Parking time is consumed only between 09:00 and 20:00.
        - A bin counts as fully occupied if the ticket covers any part of it,
          so a stay is effectively rounded up to a 10-minute multiple.
    """
    lf = data.lazy() if isinstance(data, pl.DataFrame) else data

    # Calculate raw end time
    lf = lf.with_columns(
        (pl.col("start_date") + (pl.col("paid_duration_mins") * pl.duration(minutes=1))).alias("end_time")
    )

    # Define daily operational bounds (09:00 to 20:00 of the ticket's date)
    lf = lf.with_columns([
        (pl.col("start_date").dt.date().cast(pl.Datetime) + pl.duration(hours=config.OPEN_HOUR)).alias("day_start"),
        (pl.col("start_date").dt.date().cast(pl.Datetime) + pl.duration(hours=config.CLOSE_HOUR)).alias("day_end"),
    ])

    # Apply caps to start and end times
    lf = lf.with_columns([
        pl.when(pl.col("start_date") > pl.col("day_start"))
          .then(pl.col("start_date"))
          .otherwise(pl.col("day_start"))
          .alias("start_time"),
          
        pl.when(pl.col("end_time") < pl.col("day_end"))
          .then(pl.col("end_time"))
          .otherwise(pl.col("day_end"))
          .alias("end_time"),
    ])

    # Filter out tickets completely outside operational hours (e.g., end_time <= start_time)
    lf = lf.filter(pl.col("start_time") < pl.col("end_time"))

    # Calculate first bin, last bin, and number of bins occupied
    lf = lf.with_columns([
        pl.col("start_time").dt.truncate(f"{config.INTERVAL_MINUTES}m").alias("first_bin"),
        # Subtract 1 second to handle exact boundaries (e.g., 13:20:00 becomes 13:19:59 -> bin 13:10)
        (pl.col("end_time") - pl.duration(seconds=1)).dt.truncate(f"{config.INTERVAL_MINUTES}m").alias("last_bin"),
    ])

    lf = lf.with_columns(
        ((pl.col("last_bin") - pl.col("first_bin")).dt.total_minutes() / config.INTERVAL_MINUTES + 1)
        .cast(pl.Int64)
        .alias("num_bins")
    )

    # Generate bin offsets and explode into individual bin rows
    lf = lf.with_columns(
        pl.int_ranges(start=0, end=pl.col("num_bins")).alias("bin_offsets")
    ).explode("bin_offsets")

    lf = lf.with_columns(
        (pl.col("first_bin") + (pl.col("bin_offsets") * pl.duration(minutes=config.INTERVAL_MINUTES))).alias("timestamp")
    )

    # Group by terminal and bin to count active sessions
    operational_table = (
        lf.group_by(["terminal_code", "timestamp"])
        .agg(pl.len().alias("active_sessions"))
    )

    # Get total_spaces per terminal
    capacity = (
        lf.group_by("terminal_code")
        .agg(pl.col("total_spaces").first().alias("total_spaces"))
    )

    # Join and calculate occupancy rate
    operational_table = (
        operational_table.join(capacity, on="terminal_code", how="left")
        .with_columns(
            (pl.col("active_sessions") / pl.col("total_spaces"))
            .clip(0.0, 1.0)
            .alias("occupancy_rate")
        )
        .sort(["terminal_code", "timestamp"])
    )

    operational_table = downcast_numeric(operational_table)

    return operational_table


def build_continuous_occupancy_table(
    operational_table: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """
    Build a continuous 10-minute occupancy table.

    Rules:
        - Full 24-hour grid is created: 00:00, 00:10, ..., 23:50.
        - Terminal range starts at first observed date.
        - Terminal range ends at last observed date.
        - Missing bins are filled with occupancy = 0.
        - Weekends and bank holidays are forced to occupancy = 0.
    """
    operational_table = operational_table.lazy() if isinstance(operational_table, pl.DataFrame) else operational_table

    terminal_bounds = (
        operational_table.group_by("terminal_code")
        .agg(
            min_date=pl.col("timestamp").min().dt.date(),
            max_date=pl.col("timestamp").max().dt.date(),
            total_spaces=pl.col("total_spaces").first(),
        )
        .filter(
            pl.col("min_date").is_not_null() &
            pl.col("max_date").is_not_null() &
            pl.col("total_spaces").is_not_null() &
            pl.col("total_spaces").gt(0)
        )
        .with_columns(
            n_days=(
                (pl.col("max_date") - pl.col("min_date")).dt.total_days() + 1
            ).cast(pl.Int64)
        )
        .filter(pl.col("n_days").gt(0))
    )

    terminal_dates = (
        terminal_bounds.select(
            [
                "terminal_code",
                "total_spaces",
                "min_date",
                pl.int_ranges(
                    start=0,
                    end=pl.col("n_days"),
                    dtype=pl.Int64,
                ).alias("day_offset"),
            ]
        )
        .explode("day_offset")
        .select(
            "terminal_code",
            "total_spaces",
            (
                pl.col("min_date") + pl.duration(days=pl.col("day_offset"))
            ).alias("date"),
        )
    )

    slots = (
        pl.DataFrame(
            {
                "slot_24h": list(range(config.TOTAL_BINS_PER_DAY)),
             }
        )
        .lazy()
        .with_columns(minutes_from_midnight=pl.col("slot_24h") * config.INTERVAL_MINUTES)
    )

    grid = (
        terminal_dates.join(slots, how="cross")
        .with_columns(
            timestamp=(
                pl.col("date").cast(pl.Datetime)
                + pl.duration(minutes=pl.col("minutes_from_midnight"))
            )
        )
        .select(
            "terminal_code",
            "total_spaces",
            "date",
            "slot_24h",
            "timestamp",
        )
    )

    operational_table = operational_table.select(
        ["terminal_code", "timestamp", "active_sessions", "occupancy_rate"]
    )

    continuous_table = (
        grid.join(
            operational_table,
            on=["terminal_code", "timestamp"],
            how="left",
        )
        .with_columns(
            pl.col("active_sessions").fill_null(0).cast(pl.Int64),
            pl.col("occupancy_rate").fill_null(0.0).cast(pl.Float64),
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.minute().alias("minute"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.date().alias("date"),
        )
        .with_columns(
            is_weekend=pl.col("day_of_week").is_in([6, 7]),
            is_bank_holiday=pl.col("date").is_in(config.BANK_HOLIDAYS),
        )
        .with_columns(
            is_closed_day=pl.col("is_weekend") | pl.col("is_bank_holiday")
        )
        .with_columns(
            active_sessions=pl.when(pl.col("is_closed_day"))
            .then(0)
            .otherwise(pl.col("active_sessions")),
            occupancy_rate=pl.when(pl.col("is_closed_day"))
            .then(0.0)
            .otherwise(pl.col("occupancy_rate")),
        )
        .sort(["terminal_code", "timestamp"])
    )

    continuous_table = downcast_numeric(continuous_table)

    return continuous_table