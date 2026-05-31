"""
exceptions.py
-------------
Project-wide exception hierarchy.

All custom exceptions inherit from AQEngineError so callers can catch
the entire domain with a single except clause when needed, or be granular
by catching the specific subclass.

Usage
-----
    from exceptions import GeocodingError, GEEFetchError

    try:
        geo = geocode("Pune")
    except GeocodingError as e:
        logger.error("Geocoding failed: %s", e)
"""


# ===========================================================================
# Base
# ===========================================================================

class AQEngineError(Exception):
    """
    Root exception for the Pan-India AQ Engine.
    Every custom exception in this project inherits from this class.
    """


# ===========================================================================
# Module 1 — Ingestion
# ===========================================================================

class IngestionError(AQEngineError):
    """Raised when the ingestion pipeline fails at a high level."""


class GeocodingError(IngestionError):
    """
    Raised when a location string cannot be resolved to coordinates.

    Attributes
    ----------
    query : str
        The original location string that failed.
    """
    def __init__(self, message: str, query: str = ""):
        super().__init__(message)
        self.query = query

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (query='{self.query}')" if self.query else base


class LocationOutsideIndiaError(GeocodingError):
    """
    Raised when geocoding succeeds but the result falls outside India's
    bounding box / boundary polygon.
    """
    def __init__(self, query: str, lat: float, lon: float):
        msg = (
            f"Resolved location for '{query}' is outside India: "
            f"lat={lat:.4f}, lon={lon:.4f}."
        )
        super().__init__(msg, query=query)
        self.lat = lat
        self.lon = lon


class GEEAuthError(IngestionError):
    """Raised when Google Earth Engine authentication fails."""


class GEEFetchError(IngestionError):
    """
    Raised when a GEE ImageCollection query returns no data or errors.

    Attributes
    ----------
    collection : str
        GEE dataset ID that was queried.
    """
    def __init__(self, message: str, collection: str = ""):
        super().__init__(message)
        self.collection = collection

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (collection='{self.collection}')" if self.collection else base


class WeatherFetchError(IngestionError):
    """Raised when the Open-Meteo API request fails or returns malformed data."""


class DataMergeError(IngestionError):
    """Raised when satellite and weather DataFrames cannot be aligned/merged."""


# ===========================================================================
# Module 2 — Proxy Model
# ===========================================================================

class ProxyModelError(AQEngineError):
    """Base for PM2.5 / PM10 proxy model errors."""


class ModelNotFoundError(ProxyModelError):
    """
    Raised when the trained model artifact (.joblib) is not found at
    the expected path.

    Attributes
    ----------
    path : str
        The artifact path that was checked.
    """
    def __init__(self, path: str):
        super().__init__(
            f"Model artifact not found at '{path}'. "
            "Run `python proxy_model/train.py` to generate it."
        )
        self.path = path


class InsufficientFeaturesError(ProxyModelError):
    """
    Raised when required input features (AOD, temperature, humidity) are
    missing or entirely NaN in the inference DataFrame.
    """


# ===========================================================================
# Module 3 — Database
# ===========================================================================

class DatabaseError(AQEngineError):
    """Base for all database-related errors."""


class HypertableError(DatabaseError):
    """Raised when TimescaleDB hypertable creation or insertion fails."""


class MigrationError(DatabaseError):
    """Raised when a SQL schema migration fails."""


# ===========================================================================
# Module 4 — Forecasting
# ===========================================================================

class ForecastingError(AQEngineError):
    """Base for LSTM forecasting errors."""


class SequenceTooShortError(ForecastingError):
    """
    Raised when the available historical data is shorter than the
    LSTM lookback window (168 hours).

    Attributes
    ----------
    available : int
        Number of available hourly rows.
    required : int
        Minimum rows required (LSTM_LOOKBACK_HOURS).
    """
    def __init__(self, available: int, required: int):
        super().__init__(
            f"Insufficient history for LSTM: {available} rows available, "
            f"{required} required. Ingest more data first."
        )
        self.available = available
        self.required = required


class CheckpointNotFoundError(ForecastingError):
    """Raised when the LSTM model checkpoint file is missing."""


# ===========================================================================
# Module 5 — API
# ===========================================================================

class APIError(AQEngineError):
    """Base for FastAPI layer errors."""


class InvalidRequestError(APIError):
    """Raised when an incoming API request fails validation beyond Pydantic."""


class InferenceTimeoutError(APIError):
    """Raised when an ML inference call exceeds the allowed time budget."""