import os
from datetime import date, timedelta
from pathlib import Path
from typing import Final

from src import config as src_config

APP_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = APP_DIR.parent
STATIC_DIR: Final[Path] = APP_DIR / "static"
TEMPLATES_DIR: Final[Path] = APP_DIR / "templates"


def load_dotenv(path: Path = ROOT_DIR / ".env") -> None:
    """Read KEY=VALUE lines from a gitignored .env into os.environ.

    Real environment variables win, so an exported MAPBOX_TOKEN still overrides
    the file. Kept hand-rolled to avoid a dependency for six lines of parsing.
    """
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

# Pipeline artifacts the demo reads. Both are produced by src/pipeline.py and are
# not part of the repository, so the demo needs a project that has been run.
FEATURES_PATH: Final[Path] = src_config.PROCESSED_DIR / "features.parquet"
TERMINALS_PATH: Final[Path] = src_config.ENRICHED_DIR / "clean.parquet"

# Mapbox token: from the environment or from .env, otherwise the UI prompts for it.
# A pk. token is public by design - it reaches the browser through /api/config - so
# the real protection is a URL restriction in the Mapbox dashboard, not this file.
MAPBOX_TOKEN: Final[str] = os.getenv("MAPBOX_TOKEN", "")
MAPBOX_STYLE: Final[str] = os.getenv("MAPBOX_STYLE", "")

# Dún Laoghaire-Rathdown, Dublin.
MAP_DEFAULT_CENTER: Final[list[float]] = [-6.185, 53.277]
MAP_DEFAULT_ZOOM: Final[float] = 12.1
MAP_DEFAULT_PITCH: Final[float] = 48.0
MAP_DEFAULT_BEARING: Final[float] = -12.0

# The models are trained on weekday operational bins only, so the demo offers the
# same domain: 09:00 to 19:50 in 10-minute steps.
BINS_PER_HOUR: Final[int] = 60 // src_config.INTERVAL_MINUTES
SLOT_MIN: Final[int] = src_config.OPEN_HOUR * BINS_PER_HOUR
SLOT_MAX: Final[int] = src_config.CLOSE_HOUR * BINS_PER_HOUR - 1


def slot_to_time(slot: int) -> str:
    """Render a 10-minute bin index as HH:MM."""
    return f"{slot // BINS_PER_HOUR:02d}:{(slot % BINS_PER_HOUR) * src_config.INTERVAL_MINUTES:02d}"


def previous_weekday(day: date) -> date:
    """Step back to the nearest Monday-Friday, inclusive of day itself."""
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day
