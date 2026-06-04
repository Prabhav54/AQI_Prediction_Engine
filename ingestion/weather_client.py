"""
ingestion/weather_client.py
---------------------------
Upgraded Module 1C — Unified Weather & Air Quality fetch via Open-Meteo.
Bypasses satellite dependencies completely to get real PM2.5/PM10 ground truth.
"""

import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



from config.settings import LOOKBACK_DAYS

logger = logging.getLogger(__name__)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def _fetch_open_meteo(session: requests.Session, url: str, params: dict) -> pd.DataFrame:
    try:
        resp = session.get(url, params=params, timeout=15, verify=False)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()
            
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()
    except Exception as exc:
        logger.warning(f"Failed to fetch {url}: {exc}")
        return pd.DataFrame()

def fetch_weather_and_aq(lat: float, lon: float, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    session = _build_session()
    
    # We only fetch exactly what the UI asks for (default 7 days).
    # No more downloading 365 days of data for a simple dashboard search!
    params_wx = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure,boundary_layer_height",
        "timezone": "UTC",
        "past_days": lookback_days,
        "forecast_days": 2,
    }
    
    params_aq = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth",
        "timezone": "UTC",
        "past_days": lookback_days,
        "forecast_days": 2,
    }

    logger.info(f"Fetching real Weather + AQ data for ({lat:.4f}, {lon:.4f}) over {lookback_days} days...")
    
    wx_df = _fetch_open_meteo(session, WEATHER_URL, params_wx)
    aq_df = _fetch_open_meteo(session, AQ_URL, params_aq)

    if wx_df.empty or aq_df.empty:
        raise RuntimeError("Failed to retrieve sufficient data from Open-Meteo APIs.")

    # Merge them seamlessly on the time index
    df = pd.concat([wx_df, aq_df], axis=1)
    
    # Rename variables to perfectly match the database schema
    rename_map = {
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "humidity_pct",
        "wind_speed_10m": "wind_speed_ms",
        "precipitation": "precip_mm",
        "surface_pressure": "pressure_hpa",
        "boundary_layer_height": "boundary_layer_m",
        "pm2_5": "pm25_proxy",   # <-- ADDED _proxy HERE
        "pm10": "pm10_proxy",    # <-- ADDED _proxy HERE
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3",
        "aerosol_optical_depth": "aod"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Convert wind from km/h to m/s
    if "wind_speed_ms" in df.columns:
        df["wind_speed_ms"] = df["wind_speed_ms"] / 3.6

    if "co" in df.columns:
        df["co"] = df["co"] / 1000.0    
        
    # Forward-fill minor sensor dropouts
    df = df.ffill(limit=3).bfill(limit=3)
    
    # Ensure index is correctly timezone aware and clipped to our exact window
    now_utc = datetime.now(timezone.utc)
    start_dt = pd.Timestamp(now_utc - timedelta(days=lookback_days))
    df = df.loc[(df.index >= start_dt) & (df.index <= pd.Timestamp(now_utc) + timedelta(days=2))]

    return df