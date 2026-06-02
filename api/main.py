"""
api/main.py
-----------
FastAPI application entry point.

This file wires everything together:
  - Registers the ingest and forecast routers
  - Sets up CORS so the Streamlit frontend can call the API
  - Adds a startup check to verify DB connectivity
  - Provides a /health endpoint for Docker/k8s liveness probes

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

The --reload flag is great during development — FastAPI restarts
automatically whenever you save a file. Remove it in production.
"""

import sys
from pathlib import Path

# ── Path fix — must come before ANY project imports ──────────────
# On Windows, uvicorn does not automatically add the project root
# to sys.path the way Linux does. Without this, Python can't find
# api.schemas, database.db_client, etc.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import forecast as forecast_router
from api.routes import ingest as ingest_router
from logger import get_logger

logger = get_logger(__name__)


# ================================================================
# Startup / Shutdown lifecycle
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.

    We verify DB connectivity here so the app fails loudly at startup
    rather than silently on the first real request.
    """
    logger.info("AQ Engine API starting up...")

    try:
        from sqlalchemy import text
        from database.db_client import get_sync_engine
        with get_sync_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection verified.")
    except Exception as exc:
        logger.error(
            "Database not reachable on startup: {}. "
            "Run: python database/setup_db.py",
            exc
        )
        # Don't crash the app — let /health report the issue instead

    logger.info("API ready. Docs at: http://localhost:8000/docs")
    yield

    logger.info("API shutting down.")


# ================================================================
# App initialisation
# ================================================================

app = FastAPI(
    title       = "Pan-India AQ Engine",
    description = (
        "Real-time air quality ingestion, CPCB AQI computation, "
        "and 24-hour ensemble forecasting for any Indian city.\n\n"
        "**Workflow:**\n"
        "1. `POST /ingest` — pull satellite + weather data for a location\n"
        "2. `GET /aqi` — fetch the computed CPCB AQI\n"
        "3. `GET /forecast` — get the 24-hour LSTM + XGBoost ensemble forecast"
    ),
    version     = "0.1.0",
    lifespan    = lifespan,
)


# ================================================================
# CORS — allow the Streamlit frontend to call the API
# ================================================================

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = [o.strip() for o in cors_origins],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ================================================================
# Register routers
# ================================================================

app.include_router(ingest_router.router)
app.include_router(forecast_router.router)


# ================================================================
# Health check endpoint
# ================================================================

@app.get("/health", tags=["System"])
async def health() -> dict:
    """
    Liveness probe — returns DB status and model availability.
    Used by Docker healthcheck and Streamlit to show a connection banner.
    """
    status = {"api": "ok", "db": "unknown", "models": "unknown"}

    # Check DB
    try:
        from sqlalchemy import text
        from database.db_client import get_sync_engine
        with get_sync_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        status["db"] = "ok"
    except Exception as exc:
        status["db"] = f"error: {exc}"

    # Check model artifacts
    proxy_ok    = Path("proxy_model/artifacts/xgb_pm25_proxy.joblib").exists()
    lstm_ok     = Path("forecasting/checkpoints/lstm_aqi.pt").exists()
    ensemble_ok = Path("forecasting/checkpoints/xgb_forecaster.joblib").exists()

    if proxy_ok and lstm_ok and ensemble_ok:
        status["models"] = "ready"
    else:
        missing = []
        if not proxy_ok:    missing.append("proxy model")
        if not lstm_ok:     missing.append("LSTM checkpoint")
        if not ensemble_ok: missing.append("XGBoost forecaster")
        status["models"] = f"not ready — missing: {', '.join(missing)}"

    return status


# ================================================================
# Root redirect
# ================================================================

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Pan-India AQ Engine API",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
    }