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
# NOTE: The canonical calculation lives in database/aqi_sql.sql (Module 3).
# This Python version is used for:
#   - Unit tests to validate SQL output
#   - Proxy model output verification
#   - Offline / mock runs without a DB connection

_CPCB_BREAKPOINTS: dict[str, list[tuple[float, float, float, float]]] = {
    # (Bp_low, Bp_high, I_low, I_high)
    "pm25": [
        (0,    30,   0,   50),
        (31,   60,   51,  100),
        (61,   90,   101, 200),
        (91,   120,  201, 300),
        (121,  250,  301, 400),
        (251,  500,  401, 500),
    ],
    "pm10": [
        (0,    50,   0,   50),
        (51,   100,  51,  100),
        (101,  250,  101, 200),
        (251,  350,  201, 300),
        (351,  430,  301, 400),
        (431,  600,  401, 500),
    ],
    "no2": [
        (0,    40,   0,   50),
        (41,   80,   51,  100),
        (81,   180,  101, 200),
        (181,  280,  201, 300),
        (281,  400,  301, 400),
        (401,  800,  401, 500),
    ],
    "so2": [
        (0,    40,   0,   50),
        (41,   80,   51,  100),
        (81,   380,  101, 200),
        (381,  800,  201, 300),
        (801,  1600, 301, 400),
        (1601, 2100, 401, 500),
    ],
    "co": [
        (0,    1.0,  0,   50),
        (1.1,  2.0,  51,  100),
        (2.1,  10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500),
    ],
    "o3": [
        (0,    50,   0,   50),
        (51,   100,  51,  100),
        (101,  168,  101, 200),
        (169,  208,  201, 300),
        (209,  748,  301, 400),
        (749,  1000, 401, 500),
    ],
}


def compute_sub_index(concentration: float, pollutant: str) -> Optional[float]:
    """
    Apply the CPCB linear interpolation formula for a single pollutant.

        I_p = [(I_hi - I_lo) / (Bp_hi - Bp_lo)] × (C_p - Bp_lo) + I_lo

    Parameters
    ----------
    concentration : float
        Measured or estimated concentration in CPCB units (µg/m³ for
        PM2.5/PM10/NO₂/SO₂/O₃; mg/m³ for CO).
    pollutant : str
        One of: 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3'.

    Returns
    -------
    float or None
        Sub-index value [0, 500], or None if concentration is NaN or
        exceeds the highest breakpoint.
    """
    if pd.isna(concentration) or concentration < 0:
        return None

    breakpoints = _CPCB_BREAKPOINTS.get(pollutant.lower())
    if breakpoints is None:
        raise ValueError(f"Unknown pollutant: '{pollutant}'. "
                         f"Valid: {list(_CPCB_BREAKPOINTS)}")

    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= concentration <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo

    # Above highest breakpoint → cap at 500
    if concentration > breakpoints[-1][1]:
        logger.warning(
            "Concentration %.2f for '%s' exceeds max breakpoint %.2f; capping at 500.",
            concentration, pollutant, breakpoints[-1][1],
        )
        return 500.0

    return None


def compute_aqi(pollutant_concentrations: dict[str, float]) -> dict[str, Any]:
    """
    Compute the composite AQI and prominent pollutant from a dict of
    concentrations.

    Parameters
    ----------
    pollutant_concentrations : dict
        Keys: pollutant names ('pm25', 'pm10', 'no2', 'so2', 'co', 'o3').
        Values: measured concentration in CPCB standard units.

    Returns
    -------
    dict with keys:
        aqi              : int   — final AQI (GREATEST of sub-indices)
        prominent_pollutant : str — pollutant driving the AQI
        sub_indices      : dict  — {pollutant: sub_index} for all inputs
        category         : str   — AQI category label
    """
    sub_indices: dict[str, float] = {}

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
        "prominent_pollutant": prominent.upper(),
        "sub_indices": sub_indices,
        "category": aqi_category(aqi_value),
    }


def aqi_category(aqi: int) -> str:
    """Return the CPCB AQI category label for a given AQI value."""
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
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