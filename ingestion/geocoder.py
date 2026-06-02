"""
ingestion/geocoder.py
---------------------
Module 1A — Geocoding via OpenStreetMap Nominatim.

Converts a free-text location string (e.g., "Pune, India") to a validated
(lat, lon) pair. Enforces an India bounding box to prevent off-target queries
and respects Nominatim's 1 req/s rate limit policy.
"""

import time
import logging
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import NOMINATIM_BASE_URL, NOMINATIM_USER_AGENT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# India bounding box — coarse guard against wildly off-target geocodes
# ---------------------------------------------------------------------------
INDIA_BBOX = {
    "lat_min":  6.0,
    "lat_max": 37.5,
    "lon_min": 68.0,
    "lon_max": 98.0,
}


@dataclass(frozen=True)
class GeoLocation:
    """Immutable geocoding result passed downstream to all clients."""
    query: str          # original user input
    display_name: str   # Nominatim's canonical place name
    lat: float
    lon: float
    osm_type: str       # node / way / relation
    importance: float   # Nominatim confidence score [0, 1]

    def __str__(self) -> str:
        return f"{self.display_name} ({self.lat:.4f}°N, {self.lon:.4f}°E)"


def _build_session() -> requests.Session:
    """Session with exponential-backoff retry on transient HTTP errors."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,           # waits: 0s, 1.5s, 3s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": NOMINATIM_USER_AGENT})
    return session


def _is_within_india(lat: float, lon: float) -> bool:
    return (
        INDIA_BBOX["lat_min"] <= lat <= INDIA_BBOX["lat_max"]
        and INDIA_BBOX["lon_min"] <= lon <= INDIA_BBOX["lon_max"]
    )
def geocode(lat: float, lon: float) -> str:
    return f"Location at {lat}, {lon}"


def geocode(location_query: str, country_code: str = "IN") -> GeoLocation:
    """
    Geocode a location string to a validated GeoLocation.

    Parameters
    ----------
    location_query : str
        Free-text location, e.g. "Pune", "IIT Delhi", "Connaught Place, Delhi".
    country_code : str
        ISO 3166-1 alpha-2 code to bias results. Default "IN" (India).

    Returns
    -------
    GeoLocation
        Dataclass with lat, lon, and metadata.

    Raises
    ------
    ValueError
        If no result found, or the top result falls outside India's bounding box.
    RuntimeError
        On HTTP failure after all retries.
    """
    session = _build_session()

    params = {
        "q": location_query,
        "format": "json",
        "limit": 5,
        "countrycodes": country_code,
        "addressdetails": 0,
    }

    logger.info("Geocoding query: '%s'", location_query)

    try:
        # Nominatim hard rate limit: max 1 request/second
        time.sleep(1.1)
        response = session.get(NOMINATIM_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Nominatim request failed: {exc}") from exc

    results = response.json()

    if not results:
        raise ValueError(
            f"No geocoding results found for: '{location_query}'. "
            "Try adding state/country context (e.g., 'Nagpur, Maharashtra, India')."
        )

    # Walk results in importance order; take first one inside India
    for candidate in results:
        lat = float(candidate["lat"])
        lon = float(candidate["lon"])

        if _is_within_india(lat, lon):
            geo = GeoLocation(
                query=location_query,
                display_name=candidate.get("display_name", location_query),
                lat=lat,
                lon=lon,
                osm_type=candidate.get("osm_type", "unknown"),
                importance=float(candidate.get("importance", 0.0)),
            )
            logger.info("Resolved → %s", geo)
            return geo

    # All candidates were outside India
    top = results[0]
    raise ValueError(
        f"Top geocoding result for '{location_query}' is outside India: "
        f"lat={float(top['lat']):.4f}, lon={float(top['lon']):.4f}. "
        f"Full name: {top.get('display_name', 'N/A')}"
    )