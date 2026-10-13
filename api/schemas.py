from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


# --- INGESTION SCHEMAS ---
class IngestResponse(BaseModel):
    status: str
    message: str


# --- NODE / AQI SCHEMAS ---
class NodeInfo(BaseModel):
    location_hash: str
    location_name: str
    state: Optional[str] = None
    lat: float
    lon: float
    kind: str
    distance_km: Optional[float] = None


class PollutantBreakdown(BaseModel):
    pm25_24h_avg: Optional[float] = None
    pm10_24h_avg: Optional[float] = None
    no2_24h_avg: Optional[float] = None
    so2_24h_avg: Optional[float] = None
    co_8h_avg: Optional[float] = None
    o3_8h_avg: Optional[float] = None
    sub_index_pm25: Optional[float] = None
    sub_index_pm10: Optional[float] = None
    sub_index_no2: Optional[float] = None
    sub_index_so2: Optional[float] = None
    sub_index_co: Optional[float] = None
    sub_index_o3: Optional[float] = None


class CurrentAQIResponse(BaseModel):
    node: NodeInfo
    timestamp: datetime
    aqi: int
    prominent_pollutant: str
    aqi_category: str
    pollutants: PollutantBreakdown


class MapPoint(BaseModel):
    location_hash: str
    location_name: str
    state: Optional[str] = None
    lat: float
    lon: float
    kind: str
    timestamp: datetime
    aqi: int
    aqi_category: str
    prominent_pollutant: str


# --- FORECAST SCHEMAS ---
class ForecastHour(BaseModel):
    forecast_target_time: datetime
    hours_ahead: int
    aqi: int
    aqi_lower: Optional[int] = None
    aqi_upper: Optional[int] = None
    aqi_category: str
    prominent_pollutant: Optional[str] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    aqi_lstm: Optional[int] = None
    aqi_gru: Optional[int] = None
    aqi_lgbm: Optional[int] = None


class ForecastResponse(BaseModel):
    node: NodeInfo
    generated_at: datetime
    model_version: Optional[str] = None
    hourly: List[ForecastHour]
    aqi_min: int
    aqi_max: int
    dominant_category: str
