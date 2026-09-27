"""
Read-only data access for the demo.

Everything the UI shows comes from pipeline artifacts: terminal geometry from the
enriched transactions, actual occupancy and model inputs from the feature table,
forecasts from the saved model bundles. Nothing is simulated.
"""
from collections import OrderedDict
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl

from demo import config as demo_config
from src import config as src_config
from src.models import predict

DAY_CACHE_SIZE: Final[int] = 2
FORECAST_CACHE_SIZE: Final[int] = 6


class TerminalStore:
    """Terminal coordinates, capacity and address, from the enriched transactions."""

    def __init__(self, path: Path = demo_config.TERMINALS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run src/pipeline.py before starting the demo"
            )

        self.frame = (
            pl.scan_parquet(path)
            .select(["terminal_code", "latitude", "longitude", "total_spaces", "address"])
            .unique(subset=["terminal_code"], keep="first", maintain_order=True)
            .filter(
                pl.col("latitude").is_not_null()
                & pl.col("longitude").is_not_null()
                & pl.col("total_spaces").is_not_null()
                & (pl.col("total_spaces") > 0)
            )
            .sort("terminal_code")
            .collect()
        )
        self.geometry = self.frame.select(
            ["terminal_code", "latitude", "longitude", "address"]
        )


class ModelStore:
    """Loads the saved bundles once and keeps them in memory."""

    def __init__(self, model_dir: Path = src_config.MODELS_DIR):
        self.model_dir = model_dir
        self.bundles: dict[tuple[str, str], dict[str, Any]] = {}

        if model_dir.exists():
            for path in sorted(model_dir.glob("*.joblib")):
                bundle = predict.load_bundle(path)
                self.bundles[(bundle["model_name"], bundle["horizon"])] = bundle

    @property
    def model_names(self) -> list[str]:
        return sorted({name for name, _ in self.bundles})

    @property
    def horizons(self) -> list[str]:
        """Horizons in configured order rather than alphabetically (h24h < h3h)."""
        present = {horizon for _, horizon in self.bundles}
        ordered = [h for h in src_config.HORIZONS_MINUTES if h in present]
        return ordered or sorted(present)

    def get(self, model_name: str, horizon: str) -> dict[str, Any] | None:
        return self.bundles.get((model_name, horizon))


class DayProvider:
    """Serves one historical day: actual occupancy next to a model forecast."""

    def __init__(
        self,
        terminals: TerminalStore,
        models: ModelStore,
        features_path: Path = demo_config.FEATURES_PATH,
    ):
        if not features_path.exists():
            raise FileNotFoundError(
                f"{features_path} not found - run src/pipeline.py before starting the demo"
            )

        self.terminals = terminals
        self.models = models
        self.features_path = features_path
        self._days: OrderedDict[Date, pl.DataFrame] = OrderedDict()
        self._forecasts: OrderedDict[tuple[Date, str, str], pl.DataFrame] = OrderedDict()
        self._range: tuple[Date, Date] | None = None

    def date_range(self) -> tuple[Date, Date]:
        if self._range is None:
            row = (
                pl.scan_parquet(self.features_path)
                .select(
                    earliest=pl.col("date").min(),
                    latest=pl.col("date").max(),
                )
                .collect()
                .row(0, named=True)
            )
            self._range = (row["earliest"], row["latest"])
        return self._range

    def default_date(self) -> Date:
        """A recent weekday, two days back so the 24-hour target is still defined."""
        _, latest = self.date_range()
        return demo_config.previous_weekday(latest - timedelta(days=2))

    def day(self, day: Date) -> pl.DataFrame:
        """Operational rows of one day: weekday bins between 09:00 and 19:50."""
        cached = self._days.get(day)
        if cached is not None:
            self._days.move_to_end(day)
            return cached

        frame = (
            pl.scan_parquet(self.features_path)
            .filter((pl.col("date") == day) & pl.col("is_training_origin"))
            .collect()
        )
        if frame.height == 0:
            raise ValueError(
                f"{day} has no operational rows: weekends, bank holidays and "
                "non-operational hours are outside the domain the models were trained on"
            )

        self._days[day] = frame
        self._days.move_to_end(day)
        while len(self._days) > DAY_CACHE_SIZE:
            self._days.popitem(last=False)

        return frame

    def forecast(self, day: Date, model_name: str, horizon: str) -> pl.DataFrame:
        """Model output for every operational bin of one day."""
        key = (day, model_name, horizon)
        cached = self._forecasts.get(key)
        if cached is not None:
            self._forecasts.move_to_end(key)
            return cached

        bundle = self.models.get(model_name, horizon)
        if bundle is None:
            raise ValueError(
                f"no saved bundle for model={model_name!r} horizon={horizon!r}; "
                f"available: {sorted(self.models.bundles)}"
            )

        frame = self.day(day)
        known = pl.from_pandas(bundle["terminal_map"])["terminal_code"].to_list()
        usable = frame.filter(pl.col("terminal_code").is_in(known))
        if usable.height == 0:
            raise ValueError(f"{day}: none of its terminals are known to {model_name}")

        values = predict.predict_occupancy(bundle, usable)
        result = usable.select(["terminal_code", "slot_24h"]).with_columns(
            pl.Series("predicted_occupancy", values)
        )

        self._forecasts[key] = result
        self._forecasts.move_to_end(key)
        while len(self._forecasts) > FORECAST_CACHE_SIZE:
            self._forecasts.popitem(last=False)

        return result

    def state(
        self, day: Date, slot: int, model_name: str, horizon: str
    ) -> dict[str, Any]:
        """One bin of one day: actual vs forecast for every terminal, plus totals."""
        frame = self.day(day)
        forecast = self.forecast(day, model_name, horizon)
        target = f"y_{horizon}"

        rows = (
            frame.filter(pl.col("slot_24h") == slot)
            .join(forecast, on=["terminal_code", "slot_24h"], how="inner")
            .join(self.terminals.geometry, on="terminal_code", how="inner")
            .with_columns(
                absolute_error=(pl.col("predicted_occupancy") - pl.col(target)).abs(),
                free_spaces=(
                    (1.0 - pl.col("predicted_occupancy")) * pl.col("total_spaces")
                )
                .round(0)
                .cast(pl.Int64),
            )
            .select(
                "terminal_code",
                "address",
                "latitude",
                "longitude",
                "total_spaces",
                pl.col("current_occupancy_rate").alias("occupancy_now"),
                "predicted_occupancy",
                pl.col(target).alias("actual_at_horizon"),
                "absolute_error",
                "free_spaces",
            )
            .sort("terminal_code")
        )

        if rows.height == 0:
            raise ValueError(
                f"{day} {demo_config.slot_to_time(slot)}: no rows for "
                f"model={model_name} horizon={horizon}"
            )

        summary = rows.select(
            terminals=pl.len(),
            avg_occupancy_now=pl.col("occupancy_now").mean(),
            avg_predicted=pl.col("predicted_occupancy").mean(),
            mae=pl.col("absolute_error").mean(),
            free_spaces=pl.col("free_spaces").sum(),
            congested=(pl.col("occupancy_now") >= 0.85).sum(),
        ).row(0, named=True)

        flags = frame.select(
            "is_weekend", "is_bank_holiday", "is_closed_day"
        ).row(0, named=True)

        return {
            "date": day.isoformat(),
            "slot": slot,
            "time": demo_config.slot_to_time(slot),
            "model": model_name,
            "horizon": horizon,
            **{key: bool(value) for key, value in flags.items()},
            "summary": {
                key: (round(value, 4) if isinstance(value, float) else int(value))
                for key, value in summary.items()
                if value is not None
            },
            "terminals": rows.to_dicts(),
        }