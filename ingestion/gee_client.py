"""
ingestion/gee_client.py
-----------------------
Module 1B — Google Earth Engine satellite data extraction.

Pulls tropospheric column densities from Sentinel-5P (NO₂, SO₂, CO, O₃)
and Aerosol Optical Depth (AOD) from MODIS for a point location over the
lookback window. Returns a single merged hourly DataFrame in SI units.

Authentication: uses a GEE service account JSON key (set via .env).
For local dev, `ee.Authenticate()` + `ee.Initialize()` also works.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import numpy as np

from config.settings import GEE_KEY_FILE, GEE_SERVICE_ACCOUNT, GEE_BANDS, LOOKBACK_DAYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy GEE import — only fails at runtime if earthengine-api isn't installed.
# This allows the rest of the codebase to import without GEE credentials.
# ---------------------------------------------------------------------------
def _init_gee() -> None:
    """
    Initialize GEE. Tries service account first; falls back to cached
    user credentials (useful for interactive dev with `ee.Authenticate()`).
    """
    try:
        import ee
    except ImportError as exc:
        raise ImportError(
            "earthengine-api not installed. Run: pip install earthengine-api"
        ) from exc

    if ee.data._credentials:   # already initialised in this process
        return

    if GEE_SERVICE_ACCOUNT and GEE_KEY_FILE:
        logger.info("Authenticating GEE via service account: %s", GEE_SERVICE_ACCOUNT)
        credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, GEE_KEY_FILE)
        ee.Initialize(credentials)
    else:
        logger.info("Authenticating GEE via cached user credentials.")
        ee.Initialize()


# ---------------------------------------------------------------------------
# Unit conversion factors → convert GEE native units to µg/m³ equivalents
# ---------------------------------------------------------------------------
# Sentinel-5P outputs mol/m²; we keep as mol/m² and let the proxy model
# handle the unit; AOD is dimensionless.
# These multipliers convert to more readable scales for storage:
UNIT_SCALE = {
    "NO2":  1e6,   # mol/m² → µmol/m² (better numeric range for ML)
    "SO2":  1e6,
    "CO":   1e3,   # mol/m² → mmol/m²
    "O3":   1e3,
    "AOD":  1.0,   # dimensionless
}


def _sample_image_collection(
    ee,                     # earthengine module, passed to avoid re-import
    collection_id: str,
    band: str,
    point: "ee.Geometry.Point",
    start: datetime,
    end: datetime,
    scale_m: int = 5000,    # spatial resolution for sampling (metres)
) -> pd.DataFrame:
    """
    Reduce an ImageCollection at a point to a time-series DataFrame.

    Parameters
    ----------
    collection_id : str
        GEE dataset ID.
    band : str
        Band name to extract.
    point : ee.Geometry.Point
        Sampling location.
    start, end : datetime (UTC-aware)
        Time window.
    scale_m : int
        Reducer spatial scale in metres.

    Returns
    -------
    pd.DataFrame
        Columns: [time (UTC), <band>]
    """
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(point)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .select(band)
    )

    # .getRegion() returns a list-of-lists: [header, ...rows]
    region_data = collection.getRegion(point, scale=scale_m).getInfo()

    if not region_data or len(region_data) < 2:
        logger.warning(
            "No GEE data returned for %s band '%s' in window [%s, %s]",
            collection_id, band, start.date(), end.date(),
        )
        return pd.DataFrame(columns=["time", band])

    header = region_data[0]  # ['id', 'longitude', 'latitude', 'time', <band>]
    rows   = region_data[1:]

    df = pd.DataFrame(rows, columns=header)

    # GEE timestamps are in milliseconds since epoch
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df[["time", band]].copy()
    df[band] = pd.to_numeric(df[band], errors="coerce")

    return df


def fetch_satellite_data(
    lat: float,
    lon: float,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Fetch all satellite-derived variables for a point over the lookback window.

    Pulls: NO₂, SO₂, CO, O₃ (Sentinel-5P) + AOD (MODIS).
    Each product has different overpass times so the individual series are
    resampled to 1-hour bins and outer-joined to form a unified DataFrame.

    Parameters
    ----------
    lat, lon : float
        Target coordinates.
    lookback_days : int
        Days of history to pull. Default: 7.

    Returns
    -------
    pd.DataFrame
        Hourly-ish DatetimeIndex (UTC), columns: no2, so2, co, o3, aod.
        Missing rows indicate no satellite overpass in that hour.

    Raises
    ------
    RuntimeError
        If GEE initialisation fails.
    """
    import ee

    _init_gee()

    now_utc   = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=lookback_days)

    point = ee.Geometry.Point([lon, lat])

    satellite_frames: dict[str, pd.DataFrame] = {}

    for var_name, (collection_id, band) in GEE_BANDS.items():
        logger.info("Fetching GEE %s from %s ...", var_name, collection_id)
        try:
            df = _sample_image_collection(
                ee, collection_id, band, point,
                start=start_utc, end=now_utc,
            )
            if df.empty:
                continue

            # Apply unit scaling and rename column
            col = var_name.lower()
            df = df.rename(columns={band: col})
            df[col] = df[col] * UNIT_SCALE.get(var_name, 1.0)

            # Set time as index
            df = df.set_index("time").sort_index()

            # Resample to 1H — take median across multiple overpasses
            df = df.resample("1h").median()

            satellite_frames[col] = df
            logger.info("  → %d observations for %s", df.notna().sum().sum(), var_name)

        except Exception as exc:
            logger.error("GEE fetch failed for %s: %s", var_name, exc)
            # Continue — partial data is better than a hard crash
            continue

    if not satellite_frames:
        raise RuntimeError(
            "All GEE fetches failed. Check service account credentials "
            f"and GEE dataset availability for ({lat}, {lon})."
        )

    # Outer-join all variables onto a shared hourly time index
    merged = pd.concat(satellite_frames.values(), axis=1, join="outer")
    merged.index.name = "time"
    merged = merged.sort_index()

    # Fill gaps of ≤3 hours with linear interpolation (cloud gaps are common)
    merged = merged.interpolate(method="time", limit=3)

    # Add location metadata as columns (useful for multi-city DB inserts)
    merged["lat"] = lat
    merged["lon"] = lon

    logger.info(
        "Satellite fetch complete: %d rows × %d pollutant cols, "
        "window [%s → %s]",
        len(merged), len(merged.columns) - 2,
        merged.index.min(), merged.index.max(),
    )
    return merged


# ---------------------------------------------------------------------------
# Mock fallback for development without GEE credentials
# ---------------------------------------------------------------------------
def fetch_satellite_data_mock(
    lat: float,
    lon: float,
    lookback_days: int = LOOKBACK_DAYS,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic satellite data for local dev/testing.
    Matches the exact schema of `fetch_satellite_data()`.

    Column value ranges are calibrated to typical Indian urban conditions.
    """
    rng = np.random.default_rng(seed)
    now_utc   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_utc = now_utc - timedelta(days=lookback_days)
    index     = pd.date_range(start=start_utc, end=now_utc, freq="1h", tz="UTC")
    n         = len(index)

    # Introduce NaNs to simulate cloud cover (~15% gap rate)
    def with_gaps(arr: np.ndarray, gap_rate: float = 0.15) -> np.ndarray:
        mask = rng.random(n) < gap_rate
        arr = arr.astype(float)
        arr[mask] = np.nan
        return arr

    df = pd.DataFrame(
        {
            "no2": with_gaps(rng.normal(loc=60,  scale=20,  size=n).clip(0)),   # µmol/m²
            "so2": with_gaps(rng.normal(loc=8,   scale=3,   size=n).clip(0)),   # µmol/m²
            "co":  with_gaps(rng.normal(loc=5,   scale=1.5, size=n).clip(0)),   # mmol/m²
            "o3":  with_gaps(rng.normal(loc=120, scale=30,  size=n).clip(0)),   # mmol/m²
            "aod": with_gaps(rng.uniform(low=0.1, high=0.9,  size=n)),           # dimensionless
            "lat": lat,
            "lon": lon,
        },
        index=index,
    )
    df.index.name = "time"

    logger.info(
        "[MOCK] Satellite data generated: %d rows for (%.4f, %.4f)", n, lat, lon
    )
    return df