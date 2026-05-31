"""
ingestion/pipeline.py
---------------------
Module 1 Orchestrator — ties together geocoding, satellite, and weather pulls
into a single call that returns a merged, analysis-ready DataFrame.

This is the entry point called by:
  - The CLI / batch ETL job
  - The FastAPI /ingest endpoint (via BackgroundTasks)

Merge strategy
--------------
Weather (Open-Meteo) is hourly on the dot → used as the spine.
Satellite (GEE) is resampled to 1H but has cloud gaps → left-joined onto
the weather spine so we always have a complete hourly index.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ingestion.geocoder import GeoLocation, geocode
from ingestion.weather_client import fetch_weather
from config.settings import LOOKBACK_DAYS

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    location_query: str,
    lookback_days: int = LOOKBACK_DAYS,
    use_mock_satellite: bool = False,
) -> tuple[GeoLocation, pd.DataFrame]:
    """
    Full Module 1 pipeline: geocode → fetch satellite → fetch weather → merge.

    Parameters
    ----------
    location_query : str
        Free-text location string, e.g. "Kanpur, Uttar Pradesh".
    lookback_days : int
        Historical window. Default 7 days.
    use_mock_satellite : bool
        If True, substitutes real GEE calls with synthetic data.
        Use for local dev without GEE credentials.

    Returns
    -------
    tuple[GeoLocation, pd.DataFrame]
        geo   : resolved coordinates + metadata
        df    : merged hourly DataFrame ready for the proxy model (Module 2)
                and DB ingestion (Module 3).

    DataFrame schema
    ----------------
    Index  : DatetimeIndex (UTC, hourly)
    Columns:
        Satellite  → no2, so2, co, o3, aod        (from GEE)
        Weather    → temp_c, humidity_pct,
                     wind_speed_ms, precip_mm,
                     pressure_hpa, boundary_layer_m (from Open-Meteo)
        Metadata   → lat, lon, location_name, ingested_at
    """
    ingested_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Step 1: Geocode
    # ------------------------------------------------------------------
    logger.info("=== Module 1: Ingestion Pipeline START ===")
    logger.info("Location query: '%s'", location_query)
    geo = geocode(location_query)
    logger.info("Resolved: %s", geo)

    # ------------------------------------------------------------------
    # Step 2: Satellite data
    # ------------------------------------------------------------------
    if use_mock_satellite:
        from ingestion.gee_client import fetch_satellite_data_mock
        sat_df = fetch_satellite_data_mock(geo.lat, geo.lon, lookback_days)
        logger.info("Using MOCK satellite data.")
    else:
        from ingestion.gee_client import fetch_satellite_data
        sat_df = fetch_satellite_data(geo.lat, geo.lon, lookback_days)

    # ------------------------------------------------------------------
    # Step 3: Weather data
    # ------------------------------------------------------------------
    wx_df = fetch_weather(geo.lat, geo.lon, lookback_days)

    # ------------------------------------------------------------------
    # Step 4: Merge
    # ------------------------------------------------------------------
    # Weather is the spine (no gaps). Satellite left-joined (may have NaNs).
    # Both indices are UTC DatetimeIndex already.
    merged = wx_df.join(
        sat_df.drop(columns=["lat", "lon"], errors="ignore"),
        how="left",
    )

    # Attach metadata columns (useful for multi-location DB writes)
    merged["lat"]           = geo.lat
    merged["lon"]           = geo.lon
    merged["location_name"] = geo.display_name
    merged["ingested_at"]   = ingested_at

    # Ensure index is named consistently for downstream SQL writes
    merged.index.name = "timestamp"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    sat_cols = ["no2", "so2", "co", "o3", "aod"]
    wx_cols  = ["temp_c", "humidity_pct", "wind_speed_ms", "precip_mm"]
    present  = [c for c in sat_cols + wx_cols if c in merged.columns]

    coverage = (merged[present].notna().mean() * 100).round(1)
    logger.info("Data coverage per column (%%): \n%s", coverage.to_string())
    logger.info(
        "=== Module 1 COMPLETE | rows: %d | window: [%s → %s] ===",
        len(merged),
        merged.index.min().isoformat(),
        merged.index.max().isoformat(),
    )

    return geo, merged


# ---------------------------------------------------------------------------
# CLI entry point for ad-hoc testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    parser = argparse.ArgumentParser(
        description="Run Module 1 ingestion pipeline for a location."
    )
    parser.add_argument("location", help='Location string, e.g. "Kolkata, West Bengal"')
    parser.add_argument(
        "--mock-satellite", action="store_true",
        help="Use synthetic GEE data (no credentials needed)."
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Lookback window in days (default: 7)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional CSV path to save the merged DataFrame."
    )
    args = parser.parse_args()

    try:
        geo, df = run_ingestion_pipeline(
            location_query=args.location,
            lookback_days=args.days,
            use_mock_satellite=args.mock_satellite,
        )
        print(f"\n✅ Pipeline complete for: {geo}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n{df.tail(5).to_string()}\n")

        if args.output:
            df.to_csv(args.output)
            print(f"Saved to: {args.output}")
    except (ValueError, RuntimeError) as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)