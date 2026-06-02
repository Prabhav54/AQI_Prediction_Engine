# config/__init__.py
# Re-exports the most commonly used settings so other modules can do:
#   from config import DB_URL, LOOKBACK_DAYS
# instead of:
#   from config.settings import DB_URL, LOOKBACK_DAYS

from config.settings import (
    DB_URL,
    DB_URL_SYNC,
    EE_PROJECT_ID,
    FORECAST_HOURS,
    GEE_BANDS,
    GEE_KEY_FILE,
    GEE_SERVICE_ACCOUNT,
    LOOKBACK_DAYS,
    LSTM_LOOKBACK_HOURS,
    NOMINATIM_BASE_URL,
    NOMINATIM_USER_AGENT,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_BASE_URL,
    WEATHER_HOURLY_VARS,
)

__all__ = [
    "DB_URL", "DB_URL_SYNC",
    "EE_PROJECT_ID", "GEE_SERVICE_ACCOUNT", "GEE_KEY_FILE", "GEE_BANDS",
    "OPEN_METEO_BASE_URL", "OPEN_METEO_ARCHIVE_URL", "WEATHER_HOURLY_VARS",
    "NOMINATIM_BASE_URL", "NOMINATIM_USER_AGENT",
    "LOOKBACK_DAYS", "FORECAST_HOURS", "LSTM_LOOKBACK_HOURS",
]