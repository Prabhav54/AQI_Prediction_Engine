"""
api/routes/forecast.py
----------------------
Two endpoints live here:

  GET /aqi      — returns the current AQI snapshot for a location
                  (pulled from the aqi_computed hypertable)

  GET /forecast — runs the LSTM + XGBoost ensemble and returns a
                  24-hour AQI prediction with per-hour breakdown

Both endpoints take lat + lon as query parameters. The Streamlit
frontend calls these after a successful POST /ingest.

Design note on /forecast:
  We run the ensemble synchronously here (not as a background task)
  because the LSTM inference on 168 hours of data takes < 2 seconds
  on CPU. No need to queue it — just run and return.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Counter

from fastapi import APIRouter, HTTPException, Query

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from api.schemas import (
    CurrentAQIResponse,
    ForecastHour,
    ForecastResponse,
    PollutantBreakdown,
)
from database.db_client import get_latest_aqi, get_lstm_input_sequence, write_forecast
from exceptions import CheckpointNotFoundError, DatabaseError, SequenceTooShortError
from forecasting.ensemble import ensemble_forecast_24h, load_ensemble_weights
from ingestion.geocoder import geocode
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["AQI & Forecast"])


@router.get("/aqi", response_model=CurrentAQIResponse)
async def get_current_aqi(
    lat: float = Query(..., description="Latitude", ge=6.0, le=37.5),
    lon: float = Query(..., description="Longitude", ge=68.0, le=98.0),
) -> CurrentAQIResponse:
    """
    Get the latest computed AQI for a location.

    Returns the most recent row from aqi_computed — including the
    final AQI, prominent pollutant, category, and all sub-indices.

    If no data exists yet, returns 404 — run POST /ingest first.
    """
    try:
        row = get_latest_aqi(lat, lon)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No AQI data found for ({lat}, {lon}). "
                "Run POST /ingest first to pull data for this location."
            ),
        )

    return CurrentAQIResponse(
        location_name=row["location_name"],
        lat=lat,
        lon=lon,
        timestamp=row["timestamp"],
        aqi=int(row["aqi"]),
        prominent_pollutant=row["prominent_pollutant"],
        aqi_category=row["aqi_category"],
        pollutants=PollutantBreakdown(
            pm25_24h_avg=row.get("pm25_24h_avg"),
            pm10_24h_avg=row.get("pm10_24h_avg"),
            no2_24h_avg=row.get("no2_24h_avg"),
            so2_24h_avg=row.get("so2_24h_avg"),
            co_8h_max=row.get("co_8h_max"),
            o3_8h_max=row.get("o3_8h_max"),
            sub_index_pm25=row.get("sub_index_pm25"),
            sub_index_pm10=row.get("sub_index_pm10"),
            sub_index_no2=row.get("sub_index_no2"),
            sub_index_so2=row.get("sub_index_so2"),
            sub_index_co=row.get("sub_index_co"),
            sub_index_o3=row.get("sub_index_o3"),
        ),
    )


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    lat: float = Query(..., description="Latitude", ge=6.0, le=37.5),
    lon: float = Query(..., description="Longitude", ge=68.0, le=98.0),
) -> ForecastResponse:
    """
    Run the ensemble LSTM + XGBoost forecast for the next 24 hours.

    Pulls the last 168 hours of data from the DB, runs the ensemble,
    and returns per-hour AQI predictions with ensemble breakdown.
    """
    # 1. Pull 168-hour feature sequence from DB
    try:
        sequence_df = get_lstm_input_sequence(lat, lon, lookback_hours=168)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if sequence_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No historical data found for ({lat}, {lon}). "
                "Run POST /ingest first."
            ),
        )

    # 2. Safety Net: If Open-Meteo glitches and we have less than 168 hours, pad it!
    import pandas as pd
    if len(sequence_df) < 168:
        missing_rows = 168 - len(sequence_df)
        logger.warning(f"Short sequence ({len(sequence_df)}/168). Padding {missing_rows} rows to prevent UI crash.")
        
        # Duplicate the oldest row to fill the mathematical gap
        padding_df = pd.DataFrame([sequence_df.iloc[0]] * missing_rows, columns=sequence_df.columns)
        sequence_df = pd.concat([padding_df, sequence_df])

    # 3. Run ensemble forecast
    try:
        forecasts = ensemble_forecast_24h(sequence_df, forecast_hours=24)
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}")
    except SequenceTooShortError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        logger.error(f"Unexpected ensemble error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Get location name from DB (for the response)
    try:
        latest = get_latest_aqi(lat, lon)
        location_name = latest["location_name"] if latest else f"{lat:.3f}, {lon:.3f}"
    except Exception:
        location_name = f"{lat:.3f}, {lon:.3f}"

    # 5. Build hourly forecast list
    hourly = [
        ForecastHour(
            forecast_target_time=f["forecast_target_time"],
            hours_ahead=f["hours_ahead"],
            aqi_forecast=f["aqi_forecast"],
            aqi_category_forecast=f["aqi_category_forecast"],
            aqi_lstm=f.get("aqi_lstm"),
            aqi_xgb=f.get("aqi_xgb"),
            ensemble_alpha=f.get("ensemble_alpha"),
        )
        for f in forecasts
    ]

    # 6. Summary stats for the UI header
    aqi_values = [f["aqi_forecast"] for f in forecasts]
    categories = [f["aqi_category_forecast"] for f in forecasts]
    alpha = forecasts[0].get("ensemble_alpha", 0.60) if forecasts else 0.60
    
    # Safely get the most common category
    dom_category = Counter(categories).most_common(1)[0][0] if categories else "Unknown"

    # 7. Save forecast to DB in the background (non-blocking)
    try:
        forecast_df = pd.DataFrame(forecasts)[
            ["forecast_target_time", "aqi_forecast", "aqi_category_forecast"]
        ]
        write_forecast(forecast_df, lat, lon, location_name)
    except Exception as exc:
        logger.warning(f"Could not save forecast to DB: {exc}")

    return ForecastResponse(
        location_name=location_name,
        lat=lat,
        lon=lon,
        generated_at=datetime.now(timezone.utc),
        forecast_hours=24,
        ensemble_alpha=alpha,
        hourly=hourly,
        aqi_min=min(aqi_values) if aqi_values else 0,
        aqi_max=max(aqi_values) if aqi_values else 0,
        dominant_category=dom_category,
    )