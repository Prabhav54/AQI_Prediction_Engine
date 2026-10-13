"""
config/settings.py
------------------
Single source of truth for all configuration.
Loads secrets from .env; exposes typed constants to all modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://aq_user:aq_pass@localhost:5432/air_quality_db",
)
DB_URL_SYNC: str = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://aq_user:aq_pass@localhost:5432/air_quality_db",
)

# ---------------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------------
GEE_SERVICE_ACCOUNT: str = os.getenv("GEE_SERVICE_ACCOUNT", "")
GEE_KEY_FILE: str        = os.getenv("GEE_KEY_FILE", "config/gee_key.json")
EE_PROJECT_ID: str       = os.getenv("EE_PROJECT_ID", "")   # required since GEE API v0.1.370+

# ---------------------------------------------------------------------------
# Open-Meteo (no API key required for the free tier)
# ---------------------------------------------------------------------------
OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AQ_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Free tier: 600 calls/min, 5k/hour, 10k/day. Each location in a multi-location
# request counts as one call (more for long date ranges), so we batch nodes and
# throttle on a weighted per-minute budget.
OPEN_METEO_BATCH_SIZE: int = int(os.getenv("OPEN_METEO_BATCH_SIZE", "50"))
OPEN_METEO_CALLS_PER_MIN: int = int(os.getenv("OPEN_METEO_CALLS_PER_MIN", "450"))

# ---------------------------------------------------------------------------
# Pan-India node set
# ---------------------------------------------------------------------------
# Grid nodes give uniform coverage; city nodes (config/cities.py) are added on top.
# 1.0° (~110 km) + cities ≈ 380 nodes → fits the free Open-Meteo quota when the
# pipeline runs every 3 hours. Use 0.5° (~1,150 grid nodes) with a paid key or a
# slower cadence.
GRID_RESOLUTION_DEG: float = float(os.getenv("GRID_RESOLUTION_DEG", "1.0"))
INDIA_BOUNDARY_URL: str = (
    "https://raw.githubusercontent.com/datameet/maps/master/Country/india-composite.geojson"
)

# ---------------------------------------------------------------------------
# Pollutants (model targets) — CPCB units: µg/m³ except CO in mg/m³
# ---------------------------------------------------------------------------
POLLUTANTS: list[str] = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# ---------------------------------------------------------------------------
# Retention (enforced by run_pipeline.py prune — works with or without TimescaleDB)
# ---------------------------------------------------------------------------
RAW_RETENTION_DAYS: int = int(os.getenv("RAW_RETENTION_DAYS", "120"))
FORECAST_RETENTION_DAYS: int = int(os.getenv("FORECAST_RETENTION_DAYS", "30"))

# ---------------------------------------------------------------------------
# Nominatim (OpenStreetMap geocoding — free, no key required)
# ---------------------------------------------------------------------------
NOMINATIM_BASE_URL: str = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT: str = "pan_india_aq_engine/1.0"

# ---------------------------------------------------------------------------
# Ingestion windows
# ---------------------------------------------------------------------------
LOOKBACK_DAYS: int = 8          # historical window for each ETL run (≥ 168h + buffer)
FORECAST_HOURS: int = 24        # prediction horizon
LSTM_LOOKBACK_HOURS: int = 168  # 7 days × 24h — LSTM input sequence length

# ---------------------------------------------------------------------------
# GEE dataset identifiers
# ---------------------------------------------------------------------------
SENTINEL5P_NO2 = "COPERNICUS/S5P/NRTI/L3_NO2"
SENTINEL5P_SO2 = "COPERNICUS/S5P/NRTI/L3_SO2"
SENTINEL5P_CO  = "COPERNICUS/S5P/NRTI/L3_CO"
SENTINEL5P_O3  = "COPERNICUS/S5P/NRTI/L3_O3"
MODIS_AOD      = "MODIS/061/MCD19A2_GRANULES"

# Band names we extract from each collection
GEE_BANDS = {
    "NO2":  ("COPERNICUS/S5P/NRTI/L3_NO2",  "tropospheric_NO2_column_number_density"),
    "SO2":  ("COPERNICUS/S5P/NRTI/L3_SO2",  "SO2_column_number_density"),
    "CO":   ("COPERNICUS/S5P/NRTI/L3_CO",   "CO_column_number_density"),
    "O3":   ("COPERNICUS/S5P/NRTI/L3_O3",   "O3_column_number_density"),
    "AOD":  ("MODIS/061/MCD19A2_GRANULES",   "Optical_Depth_047"),
}

# ---------------------------------------------------------------------------
# Open-Meteo variable names
# ---------------------------------------------------------------------------
WEATHER_HOURLY_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
    "boundary_layer_height",  # critical for PM dispersion
]

# ---------------------------------------------------------------------------
# CPCB AQI breakpoints (used in SQL — kept here as reference / for tests)
# ---------------------------------------------------------------------------
# Format: { pollutant: [(Bp_lo, Bp_hi, I_lo, I_hi), ...] }
# PM2.5 (µg/m³, 24-hr avg), PM10 (µg/m³), NO2 (µg/m³), SO2 (µg/m³),
# CO (mg/m³, 8-hr max), O3 (µg/m³, 8-hr max)
CPCB_BREAKPOINTS: dict = {
    "pm25": [
        (0,    30,   0,   50),
        (31,   60,   51,  100),
        (61,   90,   101, 200),
        (91,   120,  201, 300),
        (121,  250,  301, 400),
        (251,  380,  401, 500),
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
