"""
database/db_client.py
---------------------
Module 3 — Database Client (TimescaleDB)

Handles all reads and writes between the pipeline/ML code and PostgreSQL.
Nothing else in the codebase touches SQL directly — it all goes through here.

Key fix vs earlier version:
  The SQLAlchemy engines are now created lazily (on first use) instead of
  at module import time. This means importing db_client never crashes even
  if the database isn't running yet — which was causing the uvicorn startup
  error you saw. The connection is only attempted when you actually call
  a database function.
"""

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


# ================================================================
# Lazy engine initialisation
# ================================================================
# Engines are created on first use, not at import time.
# This prevents the "database not running" crash during uvicorn startup.

_async_engine  = None
_sync_engine   = None
_async_session = None


def get_async_engine():
    """Return (and lazily create) the async SQLAlchemy engine."""
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            DB_URL,
            pool_size      = 10,
            max_overflow   = 20,
            pool_timeout   = 30,
            pool_pre_ping  = True,   # test connection before using from pool
            echo           = False,
        )
    return _async_engine


def get_sync_engine():
    """Return (and lazily create) the sync SQLAlchemy engine."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            DB_URL_SYNC,
            pool_pre_ping = True,
            echo          = False,
        )
    return _sync_engine


def get_async_session_factory():
    global _async_session
    if _async_session is None:
        _async_session = sessionmaker(
            get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session


# Keep a module-level reference for the health check in main.py
# (it accesses sync_engine directly)
@property
def sync_engine():
    return get_sync_engine()


# ================================================================
# Write: Raw observations → raw_observations hypertable
# ================================================================

def write_raw_observations(df: pd.DataFrame) -> int:
    """
    Insert the merged + proxy-estimated DataFrame into raw_observations.
    Uses ON CONFLICT DO NOTHING so re-running the pipeline for the same
    location/time window never creates duplicates.

    Returns the number of new rows inserted.
    """
    if "location_hash" not in df.columns:
        df = df.copy()
        df["location_hash"] = df.apply(
            lambda r: location_hash(r["lat"], r["lon"]), axis=1
        )

    db_columns = [
        "lat", "lon", "location_name", "location_hash",
        "no2", "so2", "co", "o3", "aod",
        "temp_c", "humidity_pct", "wind_speed_ms",
        "precip_mm", "pressure_hpa", "boundary_layer_m",
        "pm25_proxy", "pm10_proxy", "ingested_at",
    ]
    available = [c for c in db_columns if c in df.columns]
    df_out    = df[available].copy()
    df_out.index.name = "timestamp"
    df_out = df_out.reset_index()

    if "ingested_at" not in df_out.columns:
        df_out["ingested_at"] = datetime.now(timezone.utc)

    engine = get_sync_engine()

    try:
        with engine.begin() as conn:
            rows_before = conn.execute(
                text("SELECT COUNT(*) FROM raw_observations")
            ).scalar()

            df_out.to_sql(
                "raw_observations_staging",
                conn,
                if_exists = "replace",
                index     = False,
                method    = "multi",
                chunksize = 500,
            )

            conn.execute(text("""
                INSERT INTO raw_observations
                SELECT * FROM raw_observations_staging
                ON CONFLICT (timestamp, location_hash) DO NOTHING
            """))
            conn.execute(text("DROP TABLE IF EXISTS raw_observations_staging"))

            rows_after = conn.execute(
                text("SELECT COUNT(*) FROM raw_observations")
            ).scalar()

        inserted = rows_after - rows_before
        logger.info("Inserted {} new rows into raw_observations.", inserted)
        return inserted

    except Exception as exc:
        raise HypertableError(f"Failed to write raw observations: {exc}") from exc


# ================================================================
# Compute: Trigger CPCB AQI SQL for a location
# ================================================================

def compute_and_store_aqi(location_hash_id: str) -> int:
    """
    Run the CPCB AQI computation for a specific location and
    materialise results into aqi_computed.

    This calls the v_aqi_final view defined in aqi_sql.sql and
    upserts the results. Re-running is always safe.
    """
    engine = get_sync_engine()

    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO aqi_computed (
                    timestamp, location_hash, location_name, lat, lon,
                    pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg,
                    co_8h_max, o3_8h_max,
                    sub_index_pm25, sub_index_pm10, sub_index_no2,
                    sub_index_so2, sub_index_co, sub_index_o3,
                    aqi, prominent_pollutant, aqi_category, computed_at
                )
                SELECT
                    timestamp, location_hash, location_name, lat, lon,
                    pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg,
                    co_8h_max, o3_8h_max,
                    sub_index_pm25, sub_index_pm10, sub_index_no2,
                    sub_index_so2, sub_index_co, sub_index_o3,
                    aqi, prominent_pollutant, aqi_category, NOW()
                FROM v_aqi_final
                WHERE location_hash = :loc_hash
                  AND aqi > 0
                ON CONFLICT (timestamp, location_hash)
                DO UPDATE SET
                    aqi                 = EXCLUDED.aqi,
                    prominent_pollutant = EXCLUDED.prominent_pollutant,
                    aqi_category        = EXCLUDED.aqi_category,
                    pm25_24h_avg        = EXCLUDED.pm25_24h_avg,
                    computed_at         = NOW()
            """), {"loc_hash": location_hash_id})

            rows = result.rowcount
            logger.info("AQI computed for {}: {} rows.", location_hash_id, rows)
            return rows

    except Exception as exc:
        raise DatabaseError(f"AQI computation failed: {exc}") from exc


# ================================================================
# Read: Latest AQI snapshot (used by GET /aqi)
# ================================================================

def get_latest_aqi(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch the most recent AQI row for a location.
    Returns None if no data exists yet for that location.
    """
    loc_hash = location_hash(lat, lon)
    engine   = get_sync_engine()

    query = text("""
        SELECT
            timestamp, location_name, aqi, prominent_pollutant,
            aqi_category, pm25_24h_avg, pm10_24h_avg, no2_24h_avg,
            so2_24h_avg, co_8h_max, o3_8h_max,
            sub_index_pm25, sub_index_pm10, sub_index_no2,
            sub_index_so2, sub_index_co, sub_index_o3
        FROM aqi_computed
        WHERE location_hash = :loc_hash
        ORDER BY timestamp DESC
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"loc_hash": loc_hash}).fetchone()
        return dict(row._mapping) if row else None
    except Exception as exc:
        raise DatabaseError(f"Failed to fetch latest AQI: {exc}") from exc


# ================================================================
# Read: 168-hour LSTM input sequence (used by GET /forecast)
# ================================================================

def get_lstm_input_sequence(
    lat: float,
    lon: float,
    lookback_hours: int = 168,
) -> pd.DataFrame:
    """
    Pull the last `lookback_hours` of engineered features for the LSTM.
    Joins aqi_computed (rolling averages) with raw_observations (weather).

    Returns empty DataFrame if insufficient data exists — caller handles this.
    """
    loc_hash = location_hash(lat, lon)
    since    = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    engine   = get_sync_engine()

    query = text("""
        SELECT
            a.timestamp,
            a.aqi, a.pm25_24h_avg, a.pm10_24h_avg,
            a.no2_24h_avg, a.so2_24h_avg, a.co_8h_max, a.o3_8h_max,
            r.temp_c, r.humidity_pct, r.wind_speed_ms,
            r.precip_mm, r.pressure_hpa, r.boundary_layer_m
        FROM aqi_computed a
        LEFT JOIN raw_observations r
            ON  a.timestamp     = r.timestamp
            AND a.location_hash = r.location_hash
        WHERE a.location_hash = :loc_hash
          AND a.timestamp     >= :since
        ORDER BY a.timestamp ASC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                query, conn,
                params      = {"loc_hash": loc_hash, "since": since},
                index_col   = "timestamp",
                parse_dates = ["timestamp"],
            )
        if df.empty:
            logger.warning(
                "No LSTM sequence data for {} (last {} hrs). Ingest first.",
                loc_hash, lookback_hours
            )
        else:
            logger.info("LSTM sequence: {} rows for {}", len(df), loc_hash)
        return df

    except Exception as exc:
        raise DatabaseError(f"Failed to fetch LSTM sequence: {exc}") from exc


# ================================================================
# Write: LSTM forecast → aqi_forecasts hypertable
# ================================================================

def write_forecast(
    forecast_df:   pd.DataFrame,
    lat:           float,
    lon:           float,
    location_name: str,
    model_version: str = "0.1.0",
) -> int:
    """
    Save the 24-hour ensemble forecast to aqi_forecasts.
    Called as a non-fatal side effect from GET /forecast.
    """
    loc_hash     = location_hash(lat, lon)
    generated_at = datetime.now(timezone.utc)
    engine       = get_sync_engine()

    df_out = forecast_df.copy()
    df_out["forecast_generated_at"] = generated_at
    df_out["location_hash"]         = loc_hash
    df_out["location_name"]         = location_name
    df_out["lat"]                   = lat
    df_out["lon"]                   = lon
    df_out["model_version"]         = model_version
    df_out["lookback_hours_used"]   = 168

    try:
        with engine.begin() as conn:
            df_out.to_sql(
                "aqi_forecasts", conn,
                if_exists = "append",
                index     = False,
                method    = "multi",
                chunksize = 24,
            )
        logger.info(
            "Wrote {} forecast rows for {} ({})",
            len(df_out), location_name,
            generated_at.strftime("%Y-%m-%dT%H:%M UTC")
        )
        return len(df_out)

    except Exception as exc:
        raise HypertableError(f"Failed to write forecast: {exc}") from exc