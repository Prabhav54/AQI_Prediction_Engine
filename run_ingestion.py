# run_ingestion.py (upgraded)
import asyncio
import aiohttp
from ingestion.grid_generator import generate_india_grid

BATCH_SIZE = 50       # concurrent API calls
RATE_LIMIT_SLEEP = 1  # seconds between batches

async def fetch_city_async(session, lat: float, lon: float, loc_hash: str):
    """Async wrapper around Open-Meteo fetch."""
    try:
        # Open-Meteo supports async natively — no geocoding needed for grid
        from ingestion.weather_client import fetch_weather_and_aq
        df = await asyncio.to_thread(fetch_weather_and_aq, lat, lon, lookback_days=1)
        df["lat"] = lat
        df["lon"] = lon
        df["location_name"] = f"{lat:.2f}N,{lon:.2f}E"
        return df
    except Exception as e:
        return None

async def ingest_all_grid_points():
    grid = generate_india_grid(resolution_deg=0.5)
    
    results = []
    for i in range(0, len(grid), BATCH_SIZE):
        batch = grid.iloc[i:i+BATCH_SIZE]
        tasks = [
            fetch_city_async(None, row.lat, row.lon, row.loc_hash)
            for _, row in batch.iterrows()
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend([r for r in batch_results if r is not None])
        await asyncio.sleep(RATE_LIMIT_SLEEP)
    
    return results