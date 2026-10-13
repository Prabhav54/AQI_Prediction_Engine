"""
ingestion/pipeline.py
---------------------
Module 1 orchestrator — one pan-India ETL pass.

    nodes (cities + grid)
      → batched Open-Meteo fetch (past LOOKBACK_DAYS + 48h forecast weather)
      → proxy-model gap fill for missing CAMS PM
      → upsert observed hours into raw_observations
      → refresh_aqi() in-database (CPCB rolling windows + sub-indices)

Returns the full fetched frame (observed + future hours) so the forecasting
step can reuse it in-memory instead of re-reading 168h of history from the DB.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from config.settings import GRID_RESOLUTION_DEG, LOOKBACK_DAYS
from database.db_client import refresh_aqi, upsert_nodes, write_raw_observations
from ingestion.grid_generator import build_node_table
from ingestion.weather_client import fetch_all_nodes, split_observed_future
from logger import get_logger
from proxy_model.predict import fill_pm_gaps
from utils import timer

logger = get_logger(__name__)


@timer
def run_pan_india_ingestion(
    resolution_deg: float = GRID_RESOLUTION_DEG,
    lookback_days: int = LOOKBACK_DAYS,
    nodes: Optional[pd.DataFrame] = None,
    write_db: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one ingestion pass.

    Returns
    -------
    (nodes, frame) — node table and long hourly frame for every node,
    covering [now - lookback_days, now + 48h].
    """
    nodes = build_node_table(resolution_deg) if nodes is None else nodes
    logger.info("Pan-India ingestion: {} nodes, {} days lookback", len(nodes), lookback_days)

    frame = fetch_all_nodes(nodes, past_days=lookback_days, forecast_days=2)
    now = datetime.now(timezone.utc)
    observed, _ = split_observed_future(frame, now)
    observed = fill_pm_gaps(observed)

    if write_db:
        upsert_nodes(nodes)
        n = write_raw_observations(observed)
        logger.info("Upserted {} observed node-hours.", n)
        refresh_aqi(since=now - timedelta(days=lookback_days))

    # Put gap-filled PM back into the frame handed to the forecaster
    frame = frame.set_index(["location_hash", "timestamp"])
    frame.update(observed.set_index(["location_hash", "timestamp"])[["pm25", "pm10"]])
    return nodes, frame.reset_index()


def run_spatial_grid_pipeline(grid_row: pd.Series, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Single-node pass (kept for ad-hoc debugging of one location)."""
    nodes = pd.DataFrame([{
        "location_hash": grid_row["location_hash"],
        "location_name": grid_row.get("location_name", ""),
        "state": grid_row.get("state"),
        "lat": float(grid_row["lat"]),
        "lon": float(grid_row["lon"]),
        "kind": grid_row.get("kind", "grid"),
    }])
    _, frame = run_pan_india_ingestion(nodes=nodes, lookback_days=lookback_days, write_db=False)
    return frame
