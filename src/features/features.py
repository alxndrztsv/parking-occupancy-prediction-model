import polars as pl

from src import config

SHORT_TERM_LAG_BINS = (1, 2, 3, 6, 12, 18)
SAME_TIME_HISTORY_DAYS = (1, 7, 14)


def build_feature_table(
    continuous_occupancy: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """Build a supervised feature table for ML and deep learning."""
    df = (
        continuous_occupancy.lazy()
        if isinstance(continuous_occupancy, pl.DataFrame)
        else continuous_occupancy
    )

    df = df.sort(["terminal_code", "timestamp"])

    # Short-term lag features
    lag_exprs = [
        pl.col("occupancy_rate")
        .shift(bins)
        .over("terminal_code")
        .alias(f"occupancy_lag_{bins}")
        for bins in SHORT_TERM_LAG_BINS
    ]

    # Same-time historical features
    same_time_exprs = [
        pl.col("occupancy_rate")
        .shift(days * config.TOTAL_BINS_PER_DAY)
        .over("terminal_code")
        .alias(f"occupancy_same_time_{days}d")
        for days in SAME_TIME_HISTORY_DAYS
    ]

    # Same-time last 7 days mean
    same_time_7d_exprs = [
        pl.col("occupancy_rate")
        .shift(days * config.TOTAL_BINS_PER_DAY)
        .over("terminal_code")
        for days in range(1, 8)
    ]

    same_time_7d_mean = pl.mean_horizontal(*same_time_7d_exprs).alias(
        "occupancy_same_time_last_7d_mean"
    )

    # Rolling features
    rolling_exprs = [
        pl.col("occupancy_rate")
        .rolling_mean(window_size=6)
        .over("terminal_code")
        .alias("occupancy_rolling_mean_1h"),
        pl.col("occupancy_rate")
        .rolling_std(window_size=6)
        .over("terminal_code")
        .alias("occupancy_rolling_std_1h"),
        pl.col("occupancy_rate")
        .rolling_mean(window_size=12)
        .over("terminal_code")
        .alias("occupancy_rolling_mean_2h"),
        pl.col("occupancy_rate")
        .rolling_mean(window_size=18)
        .over("terminal_code")
        .alias("occupancy_rolling_mean_3h"),
    ]

    # Calendar and training-origin features
    calendar_exprs = [
        pl.col("timestamp").dt.month().alias("month"),
        (
            (pl.col("hour") >= config.OPEN_HOUR)
            & (pl.col("hour") < config.CLOSE_HOUR)
            & pl.col("is_weekend").not_()
            & pl.col("is_bank_holiday").not_()
        ).alias("is_training_origin"),
    ]

    # Target features, plus the seasonal naive used as the MASE denominator
    target_exprs = []
    for horizon_name, horizon_minutes in config.HORIZONS_MINUTES.items():
        horizon_bins = horizon_minutes // config.INTERVAL_MINUTES

        target_exprs.append(
            pl.col("occupancy_rate")
            .shift(-horizon_bins)
            .over("terminal_code")
            .alias(f"y_{horizon_name}")
        )

        # The seasonal naive for occupancy(t + h) is the value one day before the
        # target moment, that is occupancy(t + h - 1 day). For h = 24h this reduces
        # to the current occupancy. The pred_ prefix keeps it out of the features.
        target_exprs.append(
            pl.col("occupancy_rate")
            .shift(config.TOTAL_BINS_PER_DAY - horizon_bins)
            .over("terminal_code")
            .alias(f"pred_seasonal_naive_{horizon_name}")
        )

    df = df.with_columns(
        lag_exprs
        + same_time_exprs
        + [same_time_7d_mean]
        + rolling_exprs
        + calendar_exprs
        + target_exprs
    )

    # Current state features
    df = df.with_columns(
        current_occupancy_rate=pl.col("occupancy_rate"),
        current_active_sessions=pl.col("active_sessions"),
        current_free_spaces=pl.when(
            pl.col("active_sessions") >= pl.col("total_spaces")
        )
        .then(0)
        .otherwise(pl.col("total_spaces") - pl.col("active_sessions")),
    )

    # Drop raw occupancy_rate/active_sessions to avoid confusion during modeling
    df = df.drop(["occupancy_rate", "active_sessions"])

    return df