"""
api/routes/ingest.py
--------------------
POST /ingest — kicks off the full data pipeline for a city.

What happens when you call this endpoint:
  1. FastAPI validates the request body (location string, lookback days)
  2. The geocoder resolves the city name to lat/lon
  3. A background task is queued — the pipeline runs without blocking the response
  4. The client immediately gets back the resolved coordinates + a status message
  5. In the background: GEE pull → weather pull → proxy model → DB write → AQI compute

Why background tasks instead of waiting?
  A full 7-day satellite + weather pull for one city takes 15-45 seconds.
  Blocking an HTTP response for that long would time out most clients.
  Instead we return instantly and the data appears in the DB asynchronously.
  The client can poll GET /aqi to check when data is ready.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from api.schemas import IngestRequest, IngestResponse
from database.db_client import compute_and_store_aqi, write_raw_observations
from exceptions import GeocodingError, IngestionError, LocationOutsideIndiaError
from ingestion.geocoder import geocode
from ingestion.pipeline import run_ingestion_pipeline
from logger import get_logger
from proxy_model.predict import run_proxy_inference
from utils import location_hash

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


def _run_pipeline_task(
    location_query:      str,
    lookback_days:       int,
    use_mock_satellite:  bool,
    loc_hash:            str,
) -> None:
    """
    The actual pipeline work — runs as a background task so the
    HTTP response isn't blocked.

    Steps:
      1. Ingestion pipeline (GEE + weather → merged DataFrame)
      2. Proxy model inference (adds PM2.5 / PM10 estimates)
      3. Write raw data to TimescaleDB
      4. Trigger CPCB AQI SQL computation
    """
    logger.info("Background pipeline started for: {}", location_query)

    try:
        # Step 1 — pull satellite + weather data and merge
        geo, merged_df = run_ingestion_pipeline(
            location_query=location_query,
            lookback_days=lookback_days,
            use_mock_satellite=use_mock_satellite,
        )

        # Step 2 — estimate PM2.5 / PM10 from AOD + weather
        merged_df = run_proxy_inference(merged_df)

        # Step 3 — write raw + proxy data to the hypertable
        rows_written = write_raw_observations(merged_df)
        logger.info("Wrote {} rows to raw_observations.", rows_written)

        # Step 4 — compute CPCB AQI and store in aqi_computed
        aqi_rows = compute_and_store_aqi(loc_hash)
        logger.info("AQI computed: {} rows in aqi_computed.", aqi_rows)

        logger.info("Pipeline complete for: {}", geo.display_name)

    except Exception as exc:
        # Log but don't re-raise — background tasks can't return HTTP errors
        logger.error(
            "Background pipeline failed for '{}': {}",
            location_query, exc
        )


@router.post("/", response_model=IngestResponse, status_code=202)
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Trigger data ingestion for a location.

    Returns HTTP 202 Accepted immediately. The pipeline runs in the
    background. Poll GET /aqi?location={location} to check when data
    is ready (usually 20–60 seconds depending on GEE response time).

    - **location**: any Indian city name ("Pune", "IIT Delhi", "Rourkela, Odisha")
    - **lookback_days**: 1–30 days of history to pull (default: 7)
    - **use_mock_satellite**: set True to skip real GEE calls (for testing)
    """
    # Geocode synchronously — this is fast (< 2s) and we need the
    # coordinates before we can queue the background task
    try:
        geo = geocode(request.location)
    except LocationOutsideIndiaError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Location is outside India: {exc}"
        )
    except GeocodingError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Could not geocode '{request.location}': {exc}"
        )

    loc_hash = location_hash(geo.lat, geo.lon)

    # Queue the pipeline as a background task
    background_tasks.add_task(
        _run_pipeline_task,
        location_query     = request.location,
        lookback_days      = request.lookback_days,
        use_mock_satellite = request.use_mock_satellite,
        loc_hash           = loc_hash,
    )

    logger.info(
        "Ingest request accepted for: {} ({:.4f}, {:.4f})",
        geo.display_name, geo.lat, geo.lon
    )

    return IngestResponse(
        status        = "accepted",
        location      = request.location,
        resolved_name = geo.display_name,
        lat           = geo.lat,
        lon           = geo.lon,
        message       = (
            f"Pipeline queued for '{geo.display_name}'. "
            f"Data will be ready in ~30–60 seconds. "
            f"Poll GET /aqi?lat={geo.lat}&lon={geo.lon} to check."
        ),
    )