from contextlib import asynccontextmanager
from datetime import date as Date

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from demo import config
from demo.service import DayProvider, ModelStore, TerminalStore
from src import config as src_config

provider: DayProvider | None = None
models: ModelStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load terminals and model bundles once, before the first request."""
    global provider, models
    models = ModelStore()
    provider = DayProvider(TerminalStore(), models)
    yield


app = FastAPI(
    title="Parking Occupancy Demo API",
    description=(
        "Historical day replay for the parking occupancy project: actual occupancy "
        "from the feature table next to the forecast of a saved model bundle."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(config.TEMPLATES_DIR / "index.html")


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "terminals": len(provider.terminals.frame) if provider else 0,
        "bundles": sorted(models.bundles) if models else [],
    }


@app.get("/api/config")
async def get_client_config():
    earliest, latest = provider.date_range()
    horizons = models.horizons
    model_names = models.model_names

    return {
        "mapbox_token": config.MAPBOX_TOKEN,
        "mapbox_style": config.MAPBOX_STYLE,
        "default_center": config.MAP_DEFAULT_CENTER,
        "default_zoom": config.MAP_DEFAULT_ZOOM,
        "default_pitch": config.MAP_DEFAULT_PITCH,
        "default_bearing": config.MAP_DEFAULT_BEARING,
        "horizons": horizons,
        "models": model_names,
        "default_horizon": horizons[0] if horizons else None,
        "default_model": (
            "lightgbm" if "lightgbm" in model_names
            else model_names[0] if model_names else None
        ),
        "date_min": earliest.isoformat(),
        "date_max": latest.isoformat(),
        "default_date": provider.default_date().isoformat(),
        "slot_min": config.SLOT_MIN,
        "slot_max": config.SLOT_MAX,
        "interval_minutes": src_config.INTERVAL_MINUTES,
        "terminals": len(provider.terminals.frame),
    }


@app.get("/api/state")
async def get_state(
    slot: int = Query(description="10-minute bin index, 54 = 09:00, 119 = 19:50"),
    day: str = Query(alias="date", description="ISO date, e.g. 2026-04-15"),
    horizon: str = Query(default="h1h"),
    model: str = Query(default="lightgbm"),
):
    if not config.SLOT_MIN <= slot <= config.SLOT_MAX:
        raise HTTPException(
            422, f"slot must be between {config.SLOT_MIN} and {config.SLOT_MAX}"
        )

    try:
        parsed = Date.fromisoformat(day)
    except ValueError:
        raise HTTPException(422, f"date must be ISO formatted, got {day!r}") from None

    try:
        return provider.state(parsed, slot, model, horizon)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None
