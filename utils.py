"""
utils.py
--------
Shared utility functions used across all modules.

Sections
--------
1. CPCB AQI calculation (Python reference implementation — SQL is canonical)
2. DataFrame validation helpers
3. Retry decorator (wraps tenacity for clean usage)
4. Unit conversion helpers
5. Date/time helpers
6. Miscellaneous
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from logger import get_logger
from exceptions import InsufficientFeaturesError

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ===========================================================================
# 1. CPCB AQI Calculation
# ===========================================================================
# Reference: CPCB National Air Quality Index, November 2014
# https://cpcb.nic.in/National-Air-Quality-Index.php
#
# NOTE: The canonical calculation lives in database/schema.sql (refresh_aqi).
# This Python mirror is used for:
#   - AQI on *forecast* hours (history + predicted concentrations)
#   - Unit tests that pin it to the SQL implementation
#   - Offline / mock runs without a DB connection

# Concentration breakpoints mapping onto AQI 0, 50, 100, 200, 300, 400, 500.
# Must match the ARRAY[...] literals in refresh_aqi().
CPCB_INDEX_LEVELS = np.array([0, 50, 100, 200, 300, 400, 500], dtype=float)
CPCB_BREAKPOINTS: dict[str, np.ndarray] = {
    "pm25": np.array([0, 30, 60, 90, 120, 250, 380], dtype=float),
    "pm10": np.array([0, 50, 100, 250, 350, 430, 600], dtype=float),
    "no2": np.array([0, 40, 80, 180, 280, 400, 800], dtype=float),
    "so2": np.array([0, 40, 80, 380, 800, 1600, 2100], dtype=float),
    "co": np.array([0, 1, 2, 10, 17, 34, 50], dtype=float),  # mg/m³
    "o3": np.array([0, 50, 100, 168, 208, 748, 1000], dtype=float),
}
# CPCB averaging windows (hours): 24h mean for PM/NO2/SO2, 8h mean for CO/O3
CPCB_WINDOWS: dict[str, int] = {"pm25": 24, "pm10": 24, "no2": 24, "so2": 24, "co": 8, "o3": 8}
_AGG_NAMES = {
    "pm25": "pm25_24h_avg", "pm10": "pm10_24h_avg", "no2": "no2_24h_avg",
    "so2": "so2_24h_avg", "co": "co_8h_avg", "o3": "o3_8h_avg",
}
_LABELS = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3"}


def cpcb_sub_index(concentration, pollutant: str):
    """
    Vectorised CPCB piecewise-linear sub-index:

        I_p = I_lo + (C_p - Bp_lo) × (I_hi - I_lo) / (Bp_hi - Bp_lo)

    Accepts a scalar or array; NaN stays NaN; values above the top
    breakpoint cap at 500 (np.interp clamps at the edges).
    """
    bp = CPCB_BREAKPOINTS.get(pollutant.lower())
    if bp is None:
        raise ValueError(f"Unknown pollutant: '{pollutant}'. Valid: {list(CPCB_BREAKPOINTS)}")
    c = np.clip(np.asarray(concentration, dtype=float), 0, None)
    out = np.interp(c, bp, CPCB_INDEX_LEVELS)
    return np.where(np.isnan(c), np.nan, out)


def compute_sub_index(concentration: float, pollutant: str) -> Optional[float]:
    """Scalar wrapper around cpcb_sub_index — returns None for missing data."""
    if concentration is None or pd.isna(concentration):
        return None
    return float(cpcb_sub_index(concentration, pollutant))


def compute_aqi(pollutant_concentrations: dict[str, float]) -> dict[str, Any]:
    """
    Composite AQI from already-averaged concentrations (24h / 8h means).

    Returns dict: aqi, prominent_pollutant, sub_indices, category.
    """
    sub_indices = {}
    for pollutant, conc in pollutant_concentrations.items():
        si = compute_sub_index(conc, pollutant)
        if si is not None:
            sub_indices[pollutant] = round(si, 2)

    if not sub_indices:
        return {
            "aqi": None,
            "prominent_pollutant": None,
            "sub_indices": {},
            "category": "Insufficient Data",
        }

    prominent = max(sub_indices, key=sub_indices.__getitem__)
    aqi_value = int(round(sub_indices[prominent]))
    return {
        "aqi": aqi_value,
        "prominent_pollutant": _LABELS[prominent],
        "sub_indices": sub_indices,
        "category": aqi_category(aqi_value),
    }


def aqi_along_paths(history: np.ndarray, forecast: np.ndarray,
                    pollutants: tuple = tuple(CPCB_WINDOWS)) -> tuple[np.ndarray, np.ndarray]:
    """
    CPCB AQI for each forecast hour of many locations at once.

    history  : (N, 23, 6) observed hourly concentrations for the 23 hours up to issue time
    forecast : (N, H, 6)  forecast hourly concentrations for t+1..t+H
    Returns (aqi (N, H) float, prominent (N, H) int index into `pollutants`).

    Each forecast hour's 24h / 8h means mix observed and forecast hours
    exactly as refresh_aqi() would once those hours are observed.
    """
    path = np.concatenate([history, forecast], axis=1)          # (N, 23 + H, 6)
    csum = np.concatenate([np.zeros_like(path[:, :1]), np.cumsum(path, axis=1)], axis=1)
    n_hist, H = history.shape[1], forecast.shape[1]
    end = np.arange(n_hist + 1, n_hist + H + 1)                 # exclusive end index per step
    subs = []
    for j, p in enumerate(pollutants):
        w = CPCB_WINDOWS[p]
        mean = (csum[:, end, j] - csum[:, end - w, j]) / w      # (N, H)
        subs.append(cpcb_sub_index(mean, p))
    subs = np.stack(subs, axis=-1)                              # (N, H, 6)
    return np.nanmax(subs, axis=-1), np.nanargmax(subs, axis=-1)


def cpcb_aqi_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    CPCB AQI for every hour of ONE location's hourly concentration series.

    `hourly` must be sorted, hourly, with columns pm25, pm10, no2, so2, co, o3.
    Rolling windows match refresh_aqi() in SQL: the current hour plus the
    previous 23 (or 7) hours.
    """
    out = pd.DataFrame(index=hourly.index)
    subs = []
    for p, w in CPCB_WINDOWS.items():
        agg = hourly[p].rolling(w, min_periods=1).mean()
        out[_AGG_NAMES[p]] = agg
        out[f"sub_index_{p}"] = cpcb_sub_index(agg.values, p)
        subs.append(out[f"sub_index_{p}"].values)
    subs = np.vstack(subs)  # (6, T)
    aqi = np.nanmax(subs, axis=0)
    out["aqi"] = np.round(aqi).astype(int)
    labels = np.array([_LABELS[p] for p in CPCB_WINDOWS])
    out["prominent_pollutant"] = labels[np.nanargmax(subs, axis=0)]
    out["aqi_category"] = [aqi_category(v) for v in out["aqi"]]
    return out


def aqi_category(aqi: int) -> str:
    """Return the CPCB AQI category label for a given AQI value."""
    for upper, label in ((50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
                         (300, "Poor"), (400, "Very Poor")):
        if aqi <= upper:
            return label
    return "Severe"


# ===========================================================================
# 2. DataFrame validation helpers
# ===========================================================================

REQUIRED_SATELLITE_COLS  = ["no2", "so2", "co", "o3", "aod"]
REQUIRED_WEATHER_COLS    = ["temp_c", "humidity_pct", "wind_speed_ms"]
REQUIRED_PROXY_INPUT_COLS = ["aod", "temp_c", "humidity_pct"]


def validate_ingestion_df(df: pd.DataFrame, min_rows: int = 24) -> None:
    """
    Assert that the merged ingestion DataFrame has the minimum structure
    needed to proceed to Module 2 (proxy model).

    Raises InsufficientFeaturesError on failure (never returns False silently).
    """
    missing_wx = [c for c in REQUIRED_WEATHER_COLS if c not in df.columns]
    if missing_wx:
        raise InsufficientFeaturesError(
            f"Weather columns missing from ingestion DataFrame: {missing_wx}"
        )

    if len(df) < min_rows:
        raise InsufficientFeaturesError(
            f"DataFrame has only {len(df)} rows; need at least {min_rows}."
        )

    # Warn (don't raise) on sparse satellite data — expected with cloud cover
    sat_coverage = {
        col: df[col].notna().mean()
        for col in REQUIRED_SATELLITE_COLS
        if col in df.columns
    }
    for col, cov in sat_coverage.items():
        if cov < 0.3:
            logger.warning(
                "Satellite column '%s' has only %.0f%% coverage — "
                "proxy model estimates will be sparse.", col, cov * 100
            )


def validate_proxy_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean proxy model inputs (AOD, temp, humidity).
    Drops rows where all three key features are simultaneously NaN.

    Returns the cleaned DataFrame.
    """
    missing = [c for c in REQUIRED_PROXY_INPUT_COLS if c not in df.columns]
    if missing:
        raise InsufficientFeaturesError(
            f"Proxy model required columns not found: {missing}. "
            "Check that the ingestion pipeline completed successfully."
        )

    all_nan_mask = df[REQUIRED_PROXY_INPUT_COLS].isna().all(axis=1)
    dropped = all_nan_mask.sum()
    if dropped > 0:
        logger.warning("Dropping %d rows with all proxy features NaN.", dropped)
        df = df[~all_nan_mask].copy()

    return df


# ===========================================================================
# 3. Retry decorator
# ===========================================================================

def with_retry(
    max_attempts: int = 3,
    wait_min: float = 1.0,
    wait_max: float = 10.0,
    reraise: bool = True,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator factory: adds exponential-backoff retry to any function.

    Parameters
    ----------
    max_attempts : int
        Total number of tries (including the first).
    wait_min, wait_max : float
        Min/max seconds between retries (exponential backoff).
    reraise : bool
        If True, re-raises the last exception after exhausting retries.
    exceptions : tuple[type[Exception], ...]
        Only retry on these exception types.

    Usage
    -----
        @with_retry(max_attempts=4, exceptions=(requests.RequestException,))
        def call_external_api(): ...
    """
    import logging as _stdlib_logging  # tenacity uses stdlib logging

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retrying = retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(min=wait_min, max=wait_max),
                retry=retry_if_exception_type(exceptions),
                reraise=reraise,
                before_sleep=before_sleep_log(
                    _stdlib_logging.getLogger(func.__module__),
                    _stdlib_logging.WARNING,
                ),
            )
            return retrying(func)(*args, **kwargs)
        return wrapper  # type: ignore[return-value]

    return decorator


# ===========================================================================
# 4. Unit conversion helpers
# ===========================================================================

def mol_per_m2_to_ug_per_m3(
    mol_per_m2: float,
    molar_mass_g_per_mol: float,
    mixing_layer_height_m: float = 1000.0,
) -> float:
    """
    Approximate conversion from Sentinel-5P column density (mol/m²)
    to near-surface concentration (µg/m³).

    This is a first-order estimate; the proxy model refines it using
    actual boundary layer height and meteorological data.

    Parameters
    ----------
    mol_per_m2 : float
        Tropospheric column density in mol/m².
    molar_mass_g_per_mol : float
        Molar mass of the gas (e.g. NO₂ = 46.0, SO₂ = 64.1, CO = 28.0).
    mixing_layer_height_m : float
        Assumed boundary layer height (default: 1000 m).

    Returns
    -------
    float
        Estimated concentration in µg/m³.
    """
    if mixing_layer_height_m <= 0:
        raise ValueError("mixing_layer_height_m must be positive.")
    # mol/m² → g/m² → g/m³ → µg/m³
    g_per_m2  = mol_per_m2 * molar_mass_g_per_mol
    g_per_m3  = g_per_m2 / mixing_layer_height_m
    ug_per_m3 = g_per_m3 * 1e6
    return ug_per_m3


# Molar masses for convenience
MOLAR_MASS = {
    "no2": 46.005,
    "so2": 64.066,
    "co":  28.010,
    "o3":  47.997,
}


# ===========================================================================
# 5. Date / time helpers
# ===========================================================================

def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def floor_to_hour(dt: datetime) -> datetime:
    """Truncate a datetime to the hour boundary."""
    return dt.replace(minute=0, second=0, microsecond=0)


def date_range_str(start: datetime, end: datetime) -> str:
    """Human-readable date range string for logging."""
    fmt = "%Y-%m-%d %H:%M UTC"
    return f"[{start.strftime(fmt)} → {end.strftime(fmt)}]"


# ===========================================================================
# 6. Miscellaneous
# ===========================================================================

def location_hash(lat: float, lon: float, precision: int = 3) -> str:
    """
    Generate a short, stable hex identifier for a (lat, lon) pair.
    Used as a cache key and for partitioning DB writes.

    Parameters
    ----------
    precision : int
        Decimal places to round to before hashing. 3 ≈ 111m resolution.
    """
    key = f"{round(lat, precision)},{round(lon, precision)}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def timer(func: F) -> F:
    """
    Decorator that logs the wall-clock execution time of a function.

    Usage
    -----
        @timer
        def slow_function(): ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug("%s completed in %.3fs", func.__qualname__, elapsed)
        return result
    return wrapper  # type: ignore[return-value]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of raising ZeroDivisionError."""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp `value` to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))
