# api/routes/forecast.py
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Counter
import pandas as pd

from fastapi import APIRouter, HTTPException, Query

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from api.schemas import CurrentAQIResponse, ForecastHour, ForecastResponse, PollutantBreakdown
from database.db_client import get_latest_aqi, get_nearest_grid_aqi, get_lstm_input_sequence, write_forecast
from exceptions import CheckpointNotFoundError, DatabaseError, SequenceTooShortError
from forecasting.ensemble import ensemble_forecast_24h
from logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["AQI & Forecast"])

@router.get("/aqi", response_model=CurrentAQIResponse)
async def get_current_aqi(
    lat: float = Query(..., description="Latitude", ge=6.0, le=37.5),
    lon: float = Query(..., description="Longitude", ge=68.0, le=98.0),
) -> CurrentAQIResponse:
    try:
        # Step 1: Query exact location signature matrix match
        row = get_latest_aqi(lat, lon)
        
        # Step 2: Fall back to PostGIS spatial indexing lookup if exact coordinate pair skips
        if row is None:
            logger.info(f"Target node index missing for ({lat}, {lon}). Initiating PostGIS spatial lookups...")
            row = get_nearest_grid_aqi(lat, lon, radius_meters=60000.0)
            
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No grid coordinates found tracking air quality data metrics within 60km of ({lat}, {lon}).",
        )

    return CurrentAQIResponse(
        location_name=row["location_name"],
        lat=lat, lon=lon, timestamp=row["timestamp"],
        aqi=int(row["aqi"]), prominent_pollutant=row["prominent_pollutant"], aqi_category=row["aqi_category"],
        pollutants=PollutantBreakdown(
            pm25_24h_avg=row.get("pm25_24h_avg"), pm10_24h_avg=row.get("pm10_24h_avg"),
            no2_24h_avg=row.get("no2_24h_avg"), so2_24h_avg=row.get("so2_24h_avg"),
            co_8h_max=row.get("co_8h_max"), o3_8h_max=row.get("o3_8h_max"),
            sub_index_pm25=row.get("sub_index_pm25"), sub_index_pm10=row.get("sub_index_pm10"),
            sub_index_no2=row.get("sub_index_no2"), sub_index_so2=row.get("sub_index_so2"),
            sub_index_co=row.get("sub_index_co"), sub_index_o3=row.get("sub_index_o3"),
        ),
    )

@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    lat: float = Query(..., description="Latitude", ge=6.0, le=37.5),
    lon: float = Query(..., description="Longitude", ge=68.0, le=98.0),
) -> ForecastResponse:
    try:
        # Step 1: Attempt extraction via precise coordinate alignment
        sequence_df = get_lstm_input_sequence(lat, lon, lookback_hours=168)
        
        # Step 2: Spatial Proximity Intersect Fallback
        # If text search produces fractional shifts, find the nearest operational tracking node hash
        if sequence_df is None or sequence_df.empty:
            logger.info(f"Historical trace index empty for ({lat}, {lon}). Searching nearest PostGIS sequence station...")
            nearest_node = get_nearest_grid_aqi(lat, lon, radius_meters=60000.0)
            
            if nearest_node and "location_hash" in nearest_node:
                # Re-extract historical sequence arrays using the closest node coordinates instead
                n_lat, n_lon = nearest_node["lat"], nearest_node["lon"]
                logger.info(f"Rerouting sequence loading matrix to neighbor node at ({n_lat}, {n_lon})")
                sequence_df = get_lstm_input_sequence(n_lat, n_lon, lookback_hours=168)

    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if sequence_df is None or sequence_df.empty:
        raise HTTPException(status_code=404, detail=f"No tracking vectors initiated near coordinates ({lat}, {lon}).")

    # Handle sequence padding logic safely
    if len(sequence_df) < 168:
        missing_rows = 168 - len(sequence_df)
        padding_df = pd.DataFrame([sequence_df.iloc[0]] * missing_rows, columns=sequence_df.columns)
        sequence_df = pd.concat([padding_df, sequence_df]).reset_index(drop=True)

    try:
        forecasts = ensemble_forecast_24h(sequence_df, forecast_hours=24)
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model checkpoints missing: {exc}")
    except SequenceTooShortError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        latest = get_latest_aqi(lat, lon) or get_nearest_grid_aqi(lat, lon, radius_meters=60000.0)
        location_name = latest["location_name"] if latest else f"{lat:.3f}, {lon:.3f}"
    except Exception:
        location_name = f"{lat:.3f}, {lon:.3f}"

    hourly = [
        ForecastHour(
            forecast_target_time=f["forecast_target_time"], hours_ahead=f["hours_ahead"],
            aqi_forecast=f["aqi_forecast"], aqi_category_forecast=f["aqi_category_forecast"],
            aqi_lstm=f.get("aqi_lstm"), aqi_xgb=f.get("aqi_xgb"), ensemble_alpha=f.get("ensemble_alpha"),
        ) for f in forecasts
    ]

    aqi_values = [f["aqi_forecast"] for f in forecasts]
    categories = [f["aqi_category_forecast"] for f in forecasts]
    dom_category = Counter(categories).most_common(1)[0][0] if categories else "Unknown"

    try:
        f_df = pd.DataFrame(forecasts)[["forecast_target_time", "aqi_forecast", "aqi_category_forecast"]]
        write_forecast(f_df, lat, lon, location_name)
    except Exception as exc:
        logger.warning(f"Background save skipped: {exc}")

    return ForecastResponse(
        location_name=location_name, lat=lat, lon=lon, generated_at=datetime.now(timezone.utc),
        forecast_hours=24, ensemble_alpha=forecasts[0].get("ensemble_alpha", 0.60),
        hourly=hourly, aqi_min=min(aqi_values), aqi_max=max(aqi_values), dominant_category=dom_category,
    )