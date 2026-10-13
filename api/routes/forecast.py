# api/routes/forecast.py
"""
Read endpoints. Forecasts are produced in batch by run_pipeline.py (every
node, every run) and stored in aqi_forecasts — requests only read them, so
latency is one indexed query instead of a model inference.
"""

import json
import math
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    CurrentAQIResponse,
    ForecastHour,
    ForecastResponse,
    MapPoint,
    NodeInfo,
    PollutantBreakdown,
)
from database.db_client import (
    latest_aqi,
    latest_aqi_all,
    latest_forecast,
    nearest_node,
    skill_summary,
)
from exceptions import DatabaseError
from logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["AQI & Forecast"])

# Read the card directly — importing forecasting.model would pull in torch
MODEL_CARD_PATH = Path(__file__).resolve().parents[2] / "forecasting" / "checkpoints" / "model_card.json"

LAT = Query(..., description="Latitude", ge=6.0, le=37.5)
LON = Query(..., description="Longitude", ge=68.0, le=98.0)


def _clean(d: dict) -> dict:
    """NaN → None so pydantic / JSON are happy."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def _node_or_404(lat: float, lon: float) -> dict:
    try:
        node = nearest_node(lat, lon)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if node is None:
        raise HTTPException(status_code=404, detail=f"No tracked node within 150 km of ({lat}, {lon}).")
    return node


@router.get("/nodes", response_model=list[MapPoint])
async def get_nodes() -> list[MapPoint]:
    """Latest AQI at every pan-India node — the map layer."""
    df = latest_aqi_all()
    return [MapPoint(**_clean(r)) for r in df.to_dict("records")]


@router.get("/aqi", response_model=CurrentAQIResponse)
async def get_current_aqi(lat: float = LAT, lon: float = LON) -> CurrentAQIResponse:
    node = _node_or_404(lat, lon)
    row = latest_aqi(node["location_hash"])
    if row is None:
        raise HTTPException(status_code=404, detail=f"No AQI yet for {node['location_name']}.")
    row = _clean(row)
    return CurrentAQIResponse(
        node=NodeInfo(**node),
        timestamp=row["timestamp"],
        aqi=int(row["aqi"]),
        prominent_pollutant=row["prominent_pollutant"],
        aqi_category=row["aqi_category"],
        pollutants=PollutantBreakdown(**{k: row.get(k) for k in PollutantBreakdown.model_fields}),
    )


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(lat: float = LAT, lon: float = LON) -> ForecastResponse:
    node = _node_or_404(lat, lon)
    df = latest_forecast(node["location_hash"])
    if df.empty:
        raise HTTPException(status_code=404,
                            detail=f"No forecast yet for {node['location_name']}. Run the pipeline.")
    hourly = [ForecastHour(**_clean(r)) for r in df.to_dict("records")]
    cats = Counter(h.aqi_category for h in hourly)
    return ForecastResponse(
        node=NodeInfo(**node),
        generated_at=df["forecast_generated_at"].iat[0],
        model_version=df["model_version"].iat[0],
        hourly=hourly,
        aqi_min=min(h.aqi for h in hourly),
        aqi_max=max(h.aqi for h in hourly),
        dominant_category=cats.most_common(1)[0][0],
    )


@router.get("/skill")
async def get_skill(days: int = Query(7, ge=1, le=30)) -> dict:
    """Offline test metrics from the model card + live skill of recent forecast runs."""
    card = json.loads(MODEL_CARD_PATH.read_text()) if MODEL_CARD_PATH.exists() else {}
    live = skill_summary(days)
    return {
        "model_version": card.get("model_version"),
        "offline_test": card.get("test", {}).get("models"),
        "live_by_horizon": json.loads(live.to_json(orient="records")),
    }
