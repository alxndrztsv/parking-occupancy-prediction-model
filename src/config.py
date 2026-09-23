from datetime import date
from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[1]

# Directories
RAW_DIR: Final[Path] = BASE_DIR / "data" / "raw"
NORMALIZED_DIR: Final[Path] = BASE_DIR / "data" / "normalized"
REFERENCE_DIR: Final[Path] = BASE_DIR / "data" / "reference"
ENRICHED_DIR: Final[Path] = BASE_DIR / "data" / "enriched"
PROCESSED_DIR: Final[Path] = BASE_DIR / "data" / "processed"
SPLITS_DIR: Final[Path] = BASE_DIR / "data" / "splits"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"
MODELS_DIR: Final[Path] = BASE_DIR / "models"

# Model hyperparameters
MODELS_CONFIG: Final[Path] = BASE_DIR / "configs" / "models.yaml"

# Coordinates
COORDS_PATH: Final[Path] = REFERENCE_DIR / "coordinates.csv"
SPACES_PATH: Final[Path] = REFERENCE_DIR / "total_spaces.csv"

RAW_GLOB: Final[str] = "*.csv"

# Timestamp formats. Parsing is explicit rather than inferred, because DD/MM and
# MM/DD are indistinguishable for days 1-12.
RAW_DATE_FORMAT = "%Y-%m-%d %H:%M:%S%.f"
NORMALIZED_DATE_FORMAT: Final[str] = "%d/%m/%Y %H:%M"
DATE_FORMAT: Final[str] = "%d/%m/%Y %H:%M"

# Time configuration
INTERVAL_MINUTES: Final[int] = 10
OPEN_HOUR: Final[int] = 9
CLOSE_HOUR: Final[int] = 20

# Full clock-day bins, needed for overnight zeros
TOTAL_BINS_PER_DAY: Final[int] = 24 * (60 // INTERVAL_MINUTES)

# Model configuration
# Forecast horizons in minutes
HORIZONS_MINUTES: Final[dict[str, int]] = {
    "h1h": 60,
    "h2h": 120,
    "h3h": 180,
    "h24h": 1440,
}
# Train/validation/test split
TEST_DAYS: Final[int] = 30
VALIDATION_DAYS: Final[int] = 60

# File names look like "jan-25.csv".
MONTH_ABBREVIATIONS: Final[dict[str, int]] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_BANK_HOLIDAYS_STRINGS: Final[list[str]] = [
    # 2024
    "2024-01-01", "2024-02-05", "2024-03-18", "2024-04-01",
    "2024-05-06", "2024-06-03", "2024-08-05", "2024-10-28", "2024-12-25", 
    "2024-12-26",
    # 2025
    "2025-01-01", "2025-02-03", "2025-03-17", "2025-04-21", "2025-05-05",
    "2025-06-02", "2025-08-04", "2025-10-27", "2025-12-25", "2025-12-26",
    # 2026
    "2026-01-01", "2026-02-02", "2026-03-17", "2026-04-06", "2026-05-04",
    "2026-06-01", "2026-08-03", "2026-10-26", "2026-12-25", "2026-12-26",
]

BANK_HOLIDAYS: Final[list[date]] = [date.fromisoformat(d) for d in _BANK_HOLIDAYS_STRINGS]

# ===== RAW SCHEMA =====
COLUMN_RENAME_MAP_RAW: Final[dict[str, str]] = {
    "PAYMENT_MEAN": "payment_mean",
    "METER_DATE": "start_date",
    "METER_CODE": "terminal_code",
    "PAID_DURATION": "paid_duration_mins",
    "TOTAL_DURATION": "total_duration_mins",
    "SYSTEM_ID": "system_id",
    "PRINTED_ID": "printed_id",
    "CARD_NAME": "card_type",
    "ZONE_DESC": "zone_desc",
    "CIRCUIT_DESC": "circuit_desc",
    "PARK_CODE": "park_code",
    "PARK_NAME": "park_name",
    "ADDRESS": "address",
    "AMOUNT": "amount",
    "END_DATE": "end_date",
    "FREE_DURATION": "free_duration_mins",
    "CURRENCY": "currency",
}

# ===== NORMALIZED SHCEMA =====
NORMALIZED_REQUIRED_COLUMNS: Final[list[str]] = [
    "terminal_code",
    "start_date",
    "address",
    "zone_desc",
    "circuit_desc",
    "park_code",
    "system_id",
    "paid_duration_mins",
    "amount",
    "payment_mean",      
]

# Column order of the normalized layer.
NORMALIZED_COLUMNS: Final[list[str]] = [
    "system_id",
    "printed_id",
    "terminal_code",
    "start_date",
    "end_date",
    "paid_duration_mins",
    "total_duration_mins",
    "free_duration_mins",
    "amount",
    "currency",
    "payment_mean",
    "card_type",
    "zone_desc",
    "circuit_desc",
    "address",
    "park_code",
    "park_name",
]

# ===== CRITICAL COLUMNS =====
# Columns that must never be null.
CRITICAL_COLUMNS: Final[list[str]] = [
    "system_id",
    "terminal_code",
    "start_date",
]

# ===== REFERENCE SCHEMA =====
COLUMN_RENAME_MAP_COORDS: Final[dict[str, str]] = {
    "Terminal Code": "terminal_code",
    "Latitude": "latitude",
    "Longitude": "longitude",
}

COLUMN_RENAME_MAP_SPACES: Final[dict[str, str]] = {
    "Terminal Code": "terminal_code",
    "Max Spaces": "total_spaces",
}

# ===== ENRICHED SCHEMA =====
ENRICHED_REQUIRED_COLUMNS: Final[list[str]] = [
    "system_id",
    "terminal_code",
    "start_date",
    "latitude",
    "longitude",
    "total_spaces",
]