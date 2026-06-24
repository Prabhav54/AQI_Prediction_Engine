# database/db_client.py
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DB_URL, DB_URL_SYNC
from exceptions import DatabaseError, HypertableError
from logger import get_logger
from utils import location_hash

logger = get_logger(__name__)

_async_engine  = None
_sync_engine   = None
_async_session = None

def get_async_engine():
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(DB_URL, pool_size=10, max_overflow=20, pool_timeout=30, pool_pre_ping=True)
    return _async_engine

def get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(DB_URL_SYNC, pool_pre_ping=True)
    return _sync_engine

def get_async_session_factory():
    global _async_session
    if _async_session is None:
        _async_session = sessionmaker(get_async_engine(), class_=AsyncSession, expire_on_commit=False)
    return _async_session

def write_spatial_grid_batch(df: pd.DataFrame) -> int:
    """
    Safely uploads a complete chunk block of raw tracking nodes straight into PostGIS.
    """
    engine = get_sync_engine()
    df_out = df.copy().reset_index()

    try:
        with engine.begin() as conn:
            df_out.to_sql("spatial_grid_staging", conn, if_exists="replace", index=False, chunksize=1000)
            
            cols = "timestamp, lat, lon, location_name, location_hash, no2, so2, co, o3, aod, temp_c, humidity_pct, wind_speed_ms, precip_mm, pressure_hpa, boundary_layer_m, pm25_proxy, pm10_proxy"
            
            conn.execute(text(f"""
                INSERT INTO raw_observations ({cols}, geom)
                SELECT {cols}, ST_SetSRID(ST_MakePoint(lon, lat), 4326)
                FROM spatial_grid_staging
                ON CONFLICT (timestamp, location_hash) DO NOTHING;
            """))
            conn.execute(text("DROP TABLE IF EXISTS spatial_grid_staging"))
        return len(df_out)
    except Exception as exc:
        raise HypertableError(f"Failed to load spatial batch to cluster: {exc}")

def get_nearest_grid_aqi(lat: float, lon: float, radius_meters: float = 60000.0) -> Optional[dict]:
    """
    Finds the closest recorded monitoring grid sequence point relative to user location.
    """
    engine = get_sync_engine()
    query = text("""
        SELECT timestamp, location_name, aqi, prominent_pollutant, aqi_category,
               pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg, co_8h_max, o3_8h_max,
               sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3
        FROM aqi_computed
        WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :radius)
        ORDER BY timestamp DESC, ST_Distance(geom, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)) ASC
        LIMIT 1;
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"lat": lat, "lon": lon, "radius": radius_meters}).fetchone()
        return dict(row._mapping) if row else None
    except Exception as exc:
        raise DatabaseError(f"Spatial proximity calculation aborted: {exc}")

def write_raw_observations(df: pd.DataFrame) -> int:
    if "location_hash" not in df.columns:
        df = df.copy()
        df["location_hash"] = df.apply(lambda r: location_hash(r["lat"], r["lon"]), axis=1)
    return write_spatial_grid_batch(df)

def compute_and_store_aqi(location_hash_id: str) -> int:
    engine = get_sync_engine()
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO aqi_computed (
                    timestamp, location_hash, location_name, lat, lon,
                    pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg, co_8h_max, o3_8h_max,
                    sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3,
                    aqi, prominent_pollutant, aqi_category, computed_at
                )
                SELECT timestamp, location_hash, location_name, lat, lon,
                       pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg, co_8h_max, o3_8h_max,
                       sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3,
                       aqi, prominent_pollutant, aqi_category, NOW()
                FROM v_aqi_final WHERE location_hash = :loc_hash AND aqi > 0
                ON CONFLICT (timestamp, location_hash) DO UPDATE SET
                    aqi=EXCLUDED.aqi, prominent_pollutant=EXCLUDED.prominent_pollutant, aqi_category=EXCLUDED.aqi_category, computed_at=NOW();
            """), {"loc_hash": location_hash_id})
            return result.rowcount
    except Exception as exc:
        raise DatabaseError(f"AQI computation failed: {exc}")

def get_latest_aqi(lat: float, lon: float) -> Optional[dict]:
    loc_hash = location_hash(lat, lon)
    engine = get_sync_engine()
    query = text("SELECT * FROM aqi_computed WHERE location_hash = :loc_hash ORDER BY timestamp DESC LIMIT 1")
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"loc_hash": loc_hash}).fetchone()
        return dict(row._mapping) if row else None
    except Exception as exc:
        raise DatabaseError(f"Failed to pull direct location row: {exc}")

def get_lstm_input_sequence(lat: float, lon: float, lookback_hours: int = 168) -> pd.DataFrame:
    loc_hash = location_hash(lat, lon)
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    engine = get_sync_engine()
    query = text("""
        SELECT a.timestamp, a.aqi, a.pm25_24h_avg, a.pm10_24h_avg, a.no2_24h_avg, a.so2_24h_avg, a.co_8h_max, a.o3_8h_max,
               r.temp_c, r.humidity_pct, r.wind_speed_ms, r.precip_mm, r.pressure_hpa, r.boundary_layer_m
        FROM aqi_computed a LEFT JOIN raw_observations r ON a.timestamp = r.timestamp AND a.location_hash = r.location_hash
        WHERE a.location_hash = :loc_hash AND a.timestamp >= :since ORDER BY a.timestamp ASC
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"loc_hash": loc_hash, "since": since}, index_col="timestamp", parse_dates=["timestamp"])
        if not df.empty:
            df['hour_of_day'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend']  = df.index.dayofweek.isin([5, 6]).astype(int)
            df['temp_change_6h'] = df['temp_c'].diff(6).fillna(0)
            df['pm25_change_3h'] = df['pm25_24h_avg'].diff(3).fillna(0)
        return df
    except Exception as exc:
        raise DatabaseError(f"Failed to fetch sequence block: {exc}")

def write_forecast(forecast_df: pd.DataFrame, lat: float, lon: float, location_name: str, model_version: str = "1.0.0") -> int:
    loc_hash = location_hash(lat, lon)
    engine = get_sync_engine()
    df_out = forecast_df.copy()
    df_out["forecast_generated_at"] = datetime.now(timezone.utc)
    df_out["location_hash"] = loc_hash
    df_out["location_name"] = location_name
    df_out["lat"], df_out["lon"] = lat, lon
    df_out["model_version"], df_out["lookback_hours_used"] = model_version, 168
    try:
        with engine.begin() as conn:
            df_out.to_sql("aqi_forecasts", conn, if_exists="append", index=False, chunksize=24)
        return len(df_out)
    except Exception as exc:
        raise HypertableError(f"Forecast storage aborted: {exc}")