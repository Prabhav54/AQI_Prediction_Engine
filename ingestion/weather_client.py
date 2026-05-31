"""
ingestion/weather_client.py
---------------------------
Module 1C — Hourly weather data via Open-Meteo.

Fetches historical + near-real-time weather for a (lat, lon) pair.
Open-Meteo's archive endpoint covers up to yesterday; the forecast endpoint
covers today + next 7 days. We stitch both to get a clean 7-day lookback
aligned with GEE satellite pulls.

No API key required for the free tier (<10,000 req/day).
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_BASE_URL,
    WEATHER_HOURLY_VARS,
    LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _parse_hourly_response(data: dict) -> pd.DataFrame:
    """
    Unpack the Open-Meteo JSON `hourly` block into a tidy DataFrame.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeTZAware (UTC), Columns: weather variables.
    """
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo response missing 'hourly' block.")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    # Rename to unambiguous internal column names
    rename_map = {
        "temperature_2m":       "temp_c",
        "relative_humidity_2m": "humidity_pct",
        "wind_speed_10m":       "wind_speed_ms",
        "precipitation":        "precip_mm",
        "surface_pressure":     "pressure_hpa",
        "boundary_layer_height":"boundary_layer_m",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Convert wind from km/h (Open-Meteo default) to m/s
    if "wind_speed_ms" in df.columns:
        df["wind_speed_ms"] = df["wind_speed_ms"] / 3.6

    return df


def fetch_weather(
    lat: float,
    lon: float,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Fetch hourly weather for the last `lookback_days` days ending NOW (UTC).

    Strategy
    --------
    - Archive endpoint  → start_date to yesterday (inclusive)
    - Forecast endpoint → today (in case archive lags by 1 day)
    Both are concatenated and deduplicated on the time index.

    Parameters
    ----------
    lat, lon : float
        Target coordinates.
    lookback_days : int
        Number of past days to fetch. Default: 7 (LOOKBACK_DAYS).

    Returns
    -------
    pd.DataFrame
        Hourly weather with UTC DatetimeIndex.
        Columns: temp_c, humidity_pct, wind_speed_ms, precip_mm,
                 pressure_hpa, boundary_layer_m (where available).

    Raises
    ------
    RuntimeError
        On HTTP failure.
    ValueError
        If the response payload is malformed.
    """
    session = _build_session()
    now_utc = datetime.now(timezone.utc)
    start_dt = now_utc - timedelta(days=lookback_days)

    start_date = start_dt.strftime("%Y-%m-%d")
    yesterday  = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    today      = now_utc.strftime("%Y-%m-%d")

    common_params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    ",".join(WEATHER_HOURLY_VARS),
        "timezone":  "UTC",
        "wind_speed_unit": "kmh",   # we convert to m/s ourselves
    }

    frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # 1. Archive pull (historical; typically available up to ~2 days ago)
    # ------------------------------------------------------------------
    archive_params = {
        **common_params,
        "start_date": start_date,
        "end_date":   yesterday,
    }
    logger.info(
        "Fetching archive weather: %s → %s for (%.4f, %.4f)",
        start_date, yesterday, lat, lon,
    )
    try:
        resp = session.get(OPEN_METEO_ARCHIVE_URL, params=archive_params, timeout=15)
        resp.raise_for_status()
        frames.append(_parse_hourly_response(resp.json()))
        logger.debug("Archive rows: %d", len(frames[-1]))
    except requests.RequestException as exc:
        raise RuntimeError(f"Open-Meteo archive request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # 2. Forecast pull (covers today; also backfills any archive lag)
    # ------------------------------------------------------------------
    forecast_params = {
        **common_params,
        "start_date": today,
        "end_date":   today,
        "past_days":  2,            # pull 2 days of hindsight from forecast model
        "forecast_days": 1,
    }
    logger.info("Fetching forecast weather for today (%s)", today)
    try:
        resp = session.get(OPEN_METEO_BASE_URL, params=forecast_params, timeout=15)
        resp.raise_for_status()
        frames.append(_parse_hourly_response(resp.json()))
        logger.debug("Forecast rows: %d", len(frames[-1]))
    except requests.RequestException as exc:
        # Non-fatal: archive data alone is sufficient for 7-day lookback
        logger.warning("Open-Meteo forecast request failed (non-fatal): %s", exc)

    if not frames:
        raise RuntimeError("No weather data retrieved from Open-Meteo.")

    # ------------------------------------------------------------------
    # Merge, deduplicate, clip to exact lookback window
    # ------------------------------------------------------------------
    df = (
        pd.concat(frames)
        .pipe(lambda d: d[~d.index.duplicated(keep="last")])   # prefer forecast over archive for overlap
        .sort_index()
    )

    # Clip to [start_dt, now_utc]
    df = df.loc[df.index >= pd.Timestamp(start_dt).tz_convert("UTC")]
    df = df.loc[df.index <= pd.Timestamp(now_utc)]

    # Forward-fill short gaps (sensor dropouts up to 2 hours)
    df = df.ffill(limit=2)

    missing_pct = df.isnull().mean() * 100
    for col, pct in missing_pct.items():
        if pct > 10:
            logger.warning("Column '%s' has %.1f%% missing values after ffill.", col, pct)

    logger.info(
        "Weather fetch complete: %d hourly rows, %d columns, "
        "window [%s → %s]",
        len(df), len(df.columns),
        df.index.min(), df.index.max(),
    )
    return df