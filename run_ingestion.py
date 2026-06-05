"""
run_ingestion.py
----------------
Automated Cron script to fetch weather and compute AQI for key Indian cities.
Run this script automatically every hour using Windows Task Scheduler or Cron.
"""

import sys
import time
from pathlib import Path

# Add project root to path so we can import internal modules
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ingestion.geocoder import geocode
from api.routes.ingest import _run_pipeline_task
from logger import get_logger

logger = get_logger(__name__)

# 50 Cities representing diverse climates across India for robust model training
TARGET_CITIES = [
    # --- Industrial & Metro (High Pollution Baseline) ---
    "Delhi", "Noida, Uttar Pradesh", "Gurugram", "Faridabad", "Ghaziabad",
    "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad",
    "Ahmedabad", "Pune", "Kanpur", "Lucknow", "Patna",

    # --- Coastal (High Humidity, Ocean Breezes) ---
    "Kochi, Kerala", "Thiruvananthapuram", "Visakhapatnam", "Puri, Odisha",
    "Goa", "Mangaluru", "Surat", "Port Blair, Andaman",

    # --- Desert & Arid (High Dust, Extreme Heat) ---
    "Jaipur", "Jodhpur", "Bikaner", "Jaisalmer", "Rajkot", 
    "Bhuj", "Gwalior",

    # --- Himalayan & High Altitude (Cold, Thin Air, Low Baseline PM) ---
    "Srinagar", "Shimla", "Manali", "Leh", "Dehradun", 
    "Gangtok", "Darjeeling", "Shillong", "Tawang",

    # --- Central & Eastern Inland (Moderate/Varied, Forested) ---
    "Bhopal", "Indore", "Nagpur", "Raipur", "Ranchi", 
    "Jamshedpur", "Guwahati", "Agartala", "Aizawl", 

    # --- Your Base Operations ---
    "Rourkela"
]

def main():
    logger.info("=== STARTING AUTOMATED HOURLY INGESTION ===")
    logger.info(f"Targeting {len(TARGET_CITIES)} cities across India.")
    
    successful = 0
    failed = 0

    for city in TARGET_CITIES:
        logger.info(f"Processing: {city}")
        try:
            # 1. Get exact coordinates
            geo = geocode(city)
            if not geo:
                logger.error(f"Could not geocode {city}. Skipping.")
                failed += 1
                continue
                
            # 2. Run the ingestion, proxy model, and AQI computation pipeline
            _run_pipeline_task(city, geo.lat, geo.lon)
            successful += 1
            
            # Rate Limiting: Sleep for 1 second between cities to avoid Open-Meteo bans
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Automated pipeline failed for {city}: {e}")
            failed += 1
            
    logger.info(f"=== HOURLY INGESTION COMPLETE | Success: {successful} | Failed: {failed} ===")

if __name__ == "__main__":
    main()