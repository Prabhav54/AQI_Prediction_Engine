from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

# --- INGESTION SCHEMAS ---
class IngestRequest(BaseModel):
    lat: float
    lon: float

class IngestResponse(BaseModel):
    status: str
    message: str
    location: Optional[str] = None

# --- FORECAST SCHEMAS ---
class PollutantBreakdown(BaseModel):
    pm25_24h_avg: Optional[float] = None
    pm10_24h_avg: Optional[float] = None
    no2_24h_avg: Optional[float] = None
    so2_24h_avg: Optional[float] = None
    co_8h_max: Optional[float] = None
    o3_8h_max: Optional[float] = None
    sub_index_pm25: Optional[int] = None
    sub_index_pm10: Optional[int] = None
    sub_index_no2: Optional[int] = None
    sub_index_so2: Optional[int] = None
    sub_index_co: Optional[int] = None
    sub_index_o3: Optional[int] = None

class CurrentAQIResponse(BaseModel):
    location_name: str
    lat: float
    lon: float
    timestamp: datetime
    aqi: int
    prominent_pollutant: str
    aqi_category: str
    pollutants: PollutantBreakdown

class ForecastHour(BaseModel):
    forecast_target_time: datetime
    hours_ahead: int
    aqi_forecast: int
    aqi_category_forecast: str
    aqi_lstm: Optional[int] = None
    aqi_xgb: Optional[int] = None
    ensemble_alpha: Optional[float] = None

class ForecastResponse(BaseModel):
    location_name: str
    lat: float
    lon: float
    generated_at: datetime
    forecast_hours: int
    ensemble_alpha: float
    hourly: List[ForecastHour]
    aqi_min: int
    aqi_max: int
    dominant_category: str