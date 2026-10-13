"""
ingestion/weather_client.py
---------------------------
Module 1 — Unified Weather + Air Quality fetch via Open-Meteo.

* Weather   : Open-Meteo forecast API (ECMWF/GFS blend) — includes the next
              48h of *forecast* weather, which the models use as known future
              covariates.
* Pollutants: Open-Meteo air-quality API (Copernicus CAMS global, ~45 km) —
              PM2.5, PM10, NO2, SO2, CO, O3, dust and aerosol optical depth.

Pan-India scale comes from multi-location requests: one HTTP call returns up
to OPEN_METEO_BATCH_SIZE nodes. A weighted per-minute throttle keeps us inside
the free-tier quota, and tenacity retries 429/5xx with exponential backoff.
"""

import math
import os
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from urllib3.util.retry import Retry

from config.settings import (
    LOOKBACK_DAYS,
    OPEN_METEO_AQ_URL,
    OPEN_METEO_BASE_URL,
    OPEN_METEO_BATCH_SIZE,
    OPEN_METEO_CALLS_PER_MIN,
)
from exceptions import WeatherFetchError
from logger import get_logger

logger = get_logger(__name__)

WEATHER_URL = OPEN_METEO_BASE_URL
HISTORICAL_WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
AQ_URL = OPEN_METEO_AQ_URL

# Open-Meteo name -> our column name
WEATHER_VARS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_dir_deg",
    "precipitation": "precip_mm",
    "surface_pressure": "pressure_hpa",
    "boundary_layer_height": "boundary_layer_m",   # critical for PM dispersion
    "shortwave_radiation": "radiation_wm2",        # drives O3 photochemistry
}
AQ_VARS = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
    "ozone": "o3",
    "dust": "dust",
    "aerosol_optical_depth": "aod",
}

WEATHER_COLS = list(WEATHER_VARS.values())
AQ_COLS = list(AQ_VARS.values())


class _RateLimitedError(Exception):
    """429 or 5xx from Open-Meteo — worth retrying after a backoff."""


# ===========================================================================
# Quota throttle
# ===========================================================================

class _CallBudget:
    """
    Sliding-window budget of *weighted* Open-Meteo calls.
    Open-Meteo counts each location as one call, multiplied up for date
    ranges longer than two weeks or more than ten variables.
    """

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        self._events: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def acquire(self, weight: float) -> None:
        weight = min(weight, self.limit)
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] > self.window_s:
                    self._events.popleft()
                used = sum(w for _, w in self._events)
                if used + weight <= self.limit:
                    self._events.append((now, weight))
                    return
                wait = self.window_s - (now - self._events[0][0]) + 0.5
            logger.info("Open-Meteo quota window full ({:.0f}/{} calls per {:.0f}s) — sleeping {:.0f}s",
                        used, self.limit, self.window_s, wait)
            time.sleep(wait)


# Free tier: 600/min and 5,000/hour — keep headroom on both
_budgets = [
    _CallBudget(OPEN_METEO_CALLS_PER_MIN, 60),
    _CallBudget(int(os.getenv("OPEN_METEO_CALLS_PER_HOUR", "4500")), 3600),
]


def _acquire(weight: float) -> None:
    for b in _budgets:
        b.acquire(weight)


def call_weight(n_locations: int, n_days: float, n_vars: int) -> float:
    """Open-Meteo's weighting: >14 days or >10 variables count as multiple calls."""
    return n_locations * max(1.0, n_days / 14) * max(1.0, n_vars / 10)


# ===========================================================================
# HTTP
# ===========================================================================

def _build_session() -> requests.Session:
    session = requests.Session()
    # urllib3 handles fast connection-level retries; tenacity handles 429/5xx
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=90),
    retry=retry_if_exception_type((_RateLimitedError, requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def _get_json(session: requests.Session, url: str, params: dict):
    resp = session.get(url, params=params, timeout=90)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _RateLimitedError(f"{resp.status_code}: {resp.text[:200]}")
    if resp.status_code != 200:
        raise WeatherFetchError(f"Open-Meteo {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _parse_hourly_response(payload: dict, rename: dict) -> pd.DataFrame:
    """One location's `hourly` block → DataFrame indexed by UTC timestamp."""
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        raise ValueError("Open-Meteo response has no hourly time axis.")
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.index.name = "timestamp"
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _time_params(past_days, forecast_days, start_date, end_date) -> tuple[dict, float]:
    if start_date is not None:
        params = {"start_date": str(start_date), "end_date": str(end_date)}
        n_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    else:
        params = {"past_days": past_days, "forecast_days": forecast_days}
        n_days = past_days + forecast_days
    return params, n_days


def _fetch_endpoint(session, url, variables: dict, lats, lons, time_params, n_days) -> list[pd.DataFrame]:
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "hourly": ",".join(variables),
        "timezone": "UTC",
        **time_params,
    }
    _acquire(call_weight(len(lats), n_days, len(variables)))
    payload = _get_json(session, url, params)
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(lats):
        raise WeatherFetchError(f"Asked for {len(lats)} locations, got {len(payloads)}.")
    return [_parse_hourly_response(p, variables) for p in payloads]


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "wind_speed_ms" in df.columns:
        df["wind_speed_ms"] = df["wind_speed_ms"] / 3.6       # km/h → m/s
    if "co" in df.columns:
        df["co"] = df["co"] / 1000.0                          # µg/m³ → mg/m³ (CPCB unit)
    # Fill short sensor/model dropouts only — long gaps stay NaN
    return df.ffill(limit=3).bfill(limit=3)


# ===========================================================================
# Public API
# ===========================================================================

def fetch_batch(
    nodes: pd.DataFrame,
    past_days: int = LOOKBACK_DAYS,
    forecast_days: int = 2,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Fetch weather + pollutants for many nodes in one pair of HTTP calls.

    Parameters
    ----------
    nodes : DataFrame with columns location_hash, lat, lon (≤ OPEN_METEO_BATCH_SIZE rows)
    past_days / forecast_days : rolling window around "now" (live mode)
    start_date / end_date     : fixed historical window (backfill mode)

    Returns
    -------
    Long DataFrame: one row per (location_hash, timestamp) with WEATHER_COLS + AQ_COLS.
    Rows after "now" hold *forecast* weather/CAMS values.
    """
    session = session or _build_session()
    tparams, n_days = _time_params(past_days, forecast_days, start_date, end_date)
    lats, lons = nodes["lat"].tolist(), nodes["lon"].tolist()

    # The live forecast API only reaches ~3 months back; use the historical
    # forecast archive (same models, back to 2022) for backfills.
    wx_url = HISTORICAL_WEATHER_URL if start_date is not None else WEATHER_URL
    wx = _fetch_endpoint(session, wx_url, WEATHER_VARS, lats, lons, tparams, n_days)
    aq = _fetch_endpoint(session, AQ_URL, AQ_VARS, lats, lons, tparams, n_days)

    frames = []
    for loc_hash, w, a in zip(nodes["location_hash"], wx, aq):
        df = _postprocess(w.join(a, how="outer"))
        df["location_hash"] = loc_hash
        frames.append(df)
    out = pd.concat(frames).reset_index()
    for col in WEATHER_COLS + AQ_COLS:
        if col not in out.columns:
            out[col] = float("nan")
    return out


def fetch_all_nodes(
    nodes: pd.DataFrame,
    batch_size: int = OPEN_METEO_BATCH_SIZE,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch every node in batches. A batch that still fails after all retries
    is logged and skipped so one bad request can't sink a pan-India run.
    """
    session = _build_session()
    frames, failed = [], 0
    n_batches = math.ceil(len(nodes) / batch_size)
    for i in range(n_batches):
        chunk = nodes.iloc[i * batch_size:(i + 1) * batch_size]
        try:
            frames.append(fetch_batch(chunk, session=session, **kwargs))
            logger.info("Open-Meteo batch {}/{} ok ({} nodes)", i + 1, n_batches, len(chunk))
        except Exception as exc:
            failed += len(chunk)
            logger.error("Open-Meteo batch {}/{} failed after retries: {}", i + 1, n_batches, exc)
    if not frames:
        raise WeatherFetchError("Every Open-Meteo batch failed.")
    if failed:
        logger.warning("{} / {} nodes skipped this run.", failed, len(nodes))
    return pd.concat(frames, ignore_index=True)


def fetch_weather_and_aq(lat: float, lon: float, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Single-location convenience wrapper (used by the Streamlit app / ad-hoc runs)."""
    from utils import location_hash

    nodes = pd.DataFrame({"location_hash": [location_hash(lat, lon)], "lat": [lat], "lon": [lon]})
    df = fetch_batch(nodes, past_days=lookback_days)
    return df.drop(columns="location_hash").set_index("timestamp")


def split_observed_future(df: pd.DataFrame, now: Optional[datetime] = None):
    """Split a fetch into observed hours (≤ now) and forecast hours (> now)."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).floor("h")
    observed = df[df["timestamp"] <= now]
    future = df[(df["timestamp"] > now) & (df["timestamp"] <= now + timedelta(hours=48))]
    return observed, future
