# ingestion/pipeline.py
import logging
from datetime import datetime, timezone
import pandas as pd

from ingestion.weather_client import fetch_weather_and_aq
from config.settings import LOOKBACK_DAYS

logger = logging.getLogger(__name__)

def run_spatial_grid_pipeline(grid_row: pd.Series, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Executes a clean data-pull sequence for an individual coordinate tracking node.
    Bypasses text geocoders entirely to enable seamless batch tasks.
    """
    ingested_at = datetime.now(timezone.utc)
    lat = float(grid_row['lat'])
    lon = float(grid_row['lon'])
    loc_hash = grid_row['loc_hash']
    loc_name = grid_row['location_name']

    logger.info(f"Processing structural grid feed for node: {loc_hash} at ({lat}, {lon})")
    
    # Target API endpoint directly via raw metrics
    merged = fetch_weather_and_aq(lat, lon, lookback_days)

    # Attach spatial metadata columns
    merged["lat"]           = lat
    merged["lon"]           = lon
    merged["location_name"] = loc_name
    merged["location_hash"] = loc_hash
    merged["ingested_at"]   = ingested_at
    merged.index.name = "timestamp"

    return merged