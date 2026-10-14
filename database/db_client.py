"""
database/db_client.py
---------------------
Module 3 — Postgres / TimescaleDB access layer.

Writes are bulk upserts via psycopg2 `execute_values` (one round-trip per
few thousand rows — matters on a serverless DB like Neon). CPCB AQI math
runs in-database through refresh_aqi() defined in schema.sql.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DB_URL_SYNC, FORECAST_RETENTION_DAYS, RAW_RETENTION_DAYS
from exceptions import DatabaseError, HypertableError
from logger import get_logger

logger = get_logger(__name__)

RAW_COLS = [
    "timestamp", "location_hash",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg", "precip_mm",
    "pressure_hpa", "boundary_layer_m", "radiation_wm2",
    "pm25", "pm10", "no2", "so2", "co", "o3", "dust", "aod", "pm_gap_filled",
]
FORECAST_COLS = [
    "forecast_generated_at", "forecast_target_time", "location_hash", "hours_ahead",
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "aqi", "aqi_lower", "aqi_upper", "aqi_category", "prominent_pollutant",
    "aqi_lstm", "aqi_gru", "aqi_lgbm", "model_version",
]

_sync_engine = None


def _normalise_url(url: str) -> str:
    # Neon / Heroku style URLs use the bare `postgres://` scheme
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    # Pin the driver: SQLAlchemy 2.1 maps bare `postgresql://` to psycopg 3, but the
    # bulk writes (execute_values) and setup_db (conn.notices) are psycopg2 APIs.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


def get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(_normalise_url(DB_URL_SYNC), pool_pre_ping=True, pool_size=5)
    return _sync_engine


def _records(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """DataFrame → list of tuples with NaN/NaT → None and numpy scalars → Python."""
    out = df[cols].astype(object).where(df[cols].notna(), None)
    return [tuple(v.item() if isinstance(v, np.generic) else v for v in row)
            for row in out.itertuples(index=False, name=None)]


def _bulk_upsert(table: str, cols: list[str], rows: list[tuple], conflict_sql: str) -> int:
    if not rows:
        return 0
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s {conflict_sql}"
    raw = get_sync_engine().raw_connection()
    try:
        with raw.cursor() as cur:
            execute_values(cur, sql, rows, page_size=5000)
        raw.commit()
        return len(rows)
    except Exception as exc:
        raw.rollback()
        raise HypertableError(f"Bulk write to {table} failed: {exc}") from exc
    finally:
        raw.close()


# ===========================================================================
# Writes
# ===========================================================================

def upsert_nodes(nodes: pd.DataFrame) -> int:
    cols = ["location_hash", "location_name", "state", "lat", "lon", "kind"]
    n = _bulk_upsert(
        "grid_nodes", cols, _records(nodes, cols),
        "ON CONFLICT (location_hash) DO UPDATE SET location_name = EXCLUDED.location_name, "
        "state = EXCLUDED.state, kind = EXCLUDED.kind",
    )
    logger.info("Registered {} grid nodes.", n)
    return n


def write_raw_observations(df: pd.DataFrame) -> int:
    """
    Upsert observed hours. CAMS revises recent hours between model runs,
    so conflicts overwrite rather than skip.
    """
    df = df.copy()
    if "pm_gap_filled" not in df.columns:
        df["pm_gap_filled"] = False
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = np.nan
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in RAW_COLS[2:])
    return _bulk_upsert(
        "raw_observations", RAW_COLS, _records(df, RAW_COLS),
        f"ON CONFLICT (timestamp, location_hash) DO UPDATE SET {updates}, ingested_at = NOW()",
    )


def refresh_aqi(since: datetime) -> int:
    """Run the in-database CPCB computation for every node-hour ≥ since."""
    try:
        with get_sync_engine().begin() as conn:
            n = conn.execute(text("SELECT refresh_aqi(:since)"), {"since": since}).scalar()
        logger.info("refresh_aqi: {} node-hours (re)computed since {}", n, since)
        return int(n or 0)
    except Exception as exc:
        raise DatabaseError(f"AQI computation failed: {exc}") from exc


def write_forecasts(df: pd.DataFrame) -> int:
    return _bulk_upsert(
        "aqi_forecasts", FORECAST_COLS, _records(df, FORECAST_COLS),
        "ON CONFLICT (forecast_generated_at, forecast_target_time, location_hash) DO NOTHING",
    )


def register_model(card: dict) -> None:
    """Upsert the deployed model card so serving layers (Vercel) can show its metrics."""
    import json

    if not card:
        return
    with get_sync_engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO model_registry (model_version, card) VALUES (:v, CAST(:c AS JSONB))
            ON CONFLICT (model_version) DO UPDATE SET card = EXCLUDED.card, last_used_at = NOW()
        """), {"v": card.get("model_version", "unknown"), "c": json.dumps(card)})


def evaluate_forecast_skill() -> int:
    with get_sync_engine().begin() as conn:
        n = conn.execute(text("SELECT evaluate_forecast_skill()")).scalar()
    logger.info("Live skill: {} (run, horizon) rows scored.", n)
    return int(n or 0)


def prune(raw_days: int = RAW_RETENTION_DAYS, forecast_days: int = FORECAST_RETENTION_DAYS) -> None:
    """Retention that works without Timescale policies (keeps Neon under its free storage cap)."""
    now = datetime.now(timezone.utc)
    with get_sync_engine().begin() as conn:
        r = conn.execute(text("DELETE FROM raw_observations WHERE timestamp < :t"),
                         {"t": now - timedelta(days=raw_days)}).rowcount
        a = conn.execute(text("DELETE FROM aqi_computed WHERE timestamp < :t"),
                         {"t": now - timedelta(days=raw_days)}).rowcount
        f = conn.execute(text("DELETE FROM aqi_forecasts WHERE forecast_generated_at < :t"),
                         {"t": now - timedelta(days=forecast_days)}).rowcount
    logger.info("Pruned {} raw, {} aqi, {} forecast rows.", r, a, f)


# ===========================================================================
# Reads
# ===========================================================================

def _read(sql: str, **params) -> pd.DataFrame:
    try:
        with get_sync_engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as exc:
        raise DatabaseError(f"Query failed: {exc}") from exc


def load_nodes() -> pd.DataFrame:
    return _read("SELECT location_hash, location_name, state, lat, lon, kind FROM grid_nodes")


def nearest_node(lat: float, lon: float, max_km: float = 150.0) -> Optional[dict]:
    """Closest tracked node (equirectangular distance — plenty for ~1k nodes)."""
    df = _read("""
        SELECT location_hash, location_name, state, lat, lon, kind,
               111.32 * SQRT(POWER(lat - :lat, 2)
                             + POWER((lon - :lon) * COS(RADIANS(:lat)), 2)) AS distance_km
        FROM grid_nodes
        ORDER BY distance_km
        LIMIT 1
    """, lat=lat, lon=lon)
    if df.empty or df["distance_km"].iat[0] > max_km:
        return None
    return df.iloc[0].to_dict()


def latest_aqi_all() -> pd.DataFrame:
    return _read("SELECT * FROM v_latest_aqi")


def latest_aqi(location_hash: str) -> Optional[dict]:
    df = _read("SELECT * FROM v_latest_aqi WHERE location_hash = :h", h=location_hash)
    return None if df.empty else df.iloc[0].to_dict()


def latest_forecast(location_hash: str) -> pd.DataFrame:
    return _read("""
        SELECT * FROM v_latest_forecast WHERE location_hash = :h ORDER BY forecast_target_time
    """, h=location_hash)


def aqi_history(location_hash: str, hours: int = 72) -> pd.DataFrame:
    return _read("""
        SELECT timestamp, aqi, aqi_category, pm25_24h_avg, pm10_24h_avg
        FROM aqi_computed
        WHERE location_hash = :h AND timestamp >= NOW() - make_interval(hours => :hours)
        ORDER BY timestamp
    """, h=location_hash, hours=hours)


def skill_summary(days: int = 7) -> pd.DataFrame:
    """Live skill per horizon, averaged over runs issued in the last `days` days."""
    return _read("""
        SELECT hours_ahead,
               SUM(n_nodes)                                         AS n,
               SUM(aqi_mae * n_nodes) / NULLIF(SUM(n_nodes), 0)     AS aqi_mae,
               SUM(persistence_aqi_mae * n_nodes) / NULLIF(SUM(n_nodes), 0) AS persistence_aqi_mae,
               SUM(category_accuracy * n_nodes) / NULLIF(SUM(n_nodes), 0)   AS category_accuracy
        FROM forecast_skill
        WHERE forecast_generated_at >= NOW() - make_interval(days => :days)
        GROUP BY hours_ahead
        ORDER BY hours_ahead
    """, days=days)


def load_observations(days: int, location_hashes: Optional[list[str]] = None) -> pd.DataFrame:
    """Hourly raw observations for training / inference, long format."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    sql = "SELECT * FROM raw_observations WHERE timestamp >= :since"
    params: dict = {"since": since}
    if location_hashes:
        sql += " AND location_hash = ANY(:hashes)"
        params["hashes"] = list(location_hashes)
    df = _read(sql + " ORDER BY location_hash, timestamp", **params)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
