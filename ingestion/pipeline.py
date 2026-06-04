"""
ingestion/pipeline.py
---------------------
Upgraded Module 1 Orchestrator
Bypasses Google Earth Engine and pulls unified Weather + Real AQI directly from Open-Meteo.
"""

import logging
from datetime import datetime, timezone
import pandas as pd

from ingestion.geocoder import GeoLocation, geocode
from ingestion.weather_client import fetch_weather_and_aq
from config.settings import LOOKBACK_DAYS

logger = logging.getLogger(__name__)

def run_ingestion_pipeline(
    location_query: str,
    lookback_days: int = LOOKBACK_DAYS,
    use_mock_satellite: bool = False, # Kept so the API doesn't crash, but ignored!
) -> tuple[GeoLocation, pd.DataFrame]:
    
    ingested_at = datetime.now(timezone.utc)

    logger.info("=== Module 1: Ingestion Pipeline START ===")
    geo = geocode(location_query)
    logger.info(f"Resolved: {geo}")

    # Step 2 & 3 Combined: Fetch both weather and REAL ground-truth AQ in one shot
    merged = fetch_weather_and_aq(geo.lat, geo.lon, lookback_days)

    # Attach metadata columns
    merged["lat"]           = geo.lat
    merged["lon"]           = geo.lon
    merged["location_name"] = geo.display_name
    merged["ingested_at"]   = ingested_at
    merged.index.name = "timestamp"

    logger.info(f"=== Module 1 COMPLETE | rows: {len(merged)} ===")
    return geo, merged


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    parser = argparse.ArgumentParser(description="Run Module 1 ingestion pipeline.")
    parser.add_argument("location", help='Location string, e.g. "Goa"')
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7).")
    args = parser.parse_args()

    try:
        from database.db_client import write_raw_observations
        
        # Now passing exactly the positional arguments needed!
        geo, df = run_ingestion_pipeline(args.location, args.days)
        
        print(f"\n✅ Pipeline complete for: {geo}")
        print("\nSaving data to TimescaleDB...")
        inserted = write_raw_observations(df)
        print(f"✅ Successfully inserted {inserted} rows into the database!\n")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)