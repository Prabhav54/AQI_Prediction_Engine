"""
tests/test_geocoder.py
----------------------
Unit tests for Module 1A (geocoder) and Module 1C (weather client).
Run: pytest tests/ -v

GEE tests are skipped by default (require credentials); use --run-gee flag.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from ingestion.geocoder import geocode, GeoLocation, _is_within_india
from ingestion.weather_client import _parse_hourly_response


# ===========================================================================
# Geocoder tests
# ===========================================================================

class TestIsWithinIndia:
    def test_delhi_is_inside(self):
        assert _is_within_india(28.6139, 77.2090) is True

    def test_london_is_outside(self):
        assert _is_within_india(51.5074, -0.1278) is False

    def test_boundary_south_tip(self):
        # Kanyakumari ~8.08°N — inside
        assert _is_within_india(8.08, 77.55) is True

    def test_boundary_north(self):
        # Leh, Ladakh ~34.15°N — inside
        assert _is_within_india(34.15, 77.58) is True

    def test_pakistan_outside(self):
        # Karachi (24.86°N, 67.01°E) — clearly west of India bbox lon_min=68.0
        # Note: a coarse rect bbox cannot exclude all of Pakistan; proper geo-fencing
        # needs a shapefile. This test verifies the lon_min guard works.
        assert _is_within_india(24.86, 67.01) is False  # Karachi


class TestGeocode:
    """Uses mock HTTP responses to avoid network calls in CI."""

    _mock_response_pune = [
        {
            "lat": "18.5204",
            "lon": "73.8567",
            "display_name": "Pune, Pune District, Maharashtra, India",
            "osm_type": "relation",
            "importance": 0.7542,
        }
    ]

    _mock_response_empty = []

    _mock_response_outside_india = [
        {
            "lat": "51.5074",
            "lon": "-0.1278",
            "display_name": "London, England, United Kingdom",
            "osm_type": "relation",
            "importance": 0.95,
        }
    ]

    def _make_mock_response(self, json_data: list) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    @patch("ingestion.geocoder.time.sleep")  # skip the 1.1s rate-limit sleep
    @patch("ingestion.geocoder.requests.Session.get")
    def test_geocode_pune_success(self, mock_get, mock_sleep):
        mock_get.return_value = self._make_mock_response(self._mock_response_pune)
        result = geocode("Pune")

        assert isinstance(result, GeoLocation)
        assert abs(result.lat - 18.5204) < 1e-3
        assert abs(result.lon - 73.8567) < 1e-3
        assert "Pune" in result.display_name

    @patch("ingestion.geocoder.time.sleep")
    @patch("ingestion.geocoder.requests.Session.get")
    def test_geocode_empty_raises_valueerror(self, mock_get, mock_sleep):
        mock_get.return_value = self._make_mock_response(self._mock_response_empty)
        with pytest.raises(ValueError, match="No geocoding results"):
            geocode("XYZ_NONEXISTENT_PLACE_12345")

    @patch("ingestion.geocoder.time.sleep")
    @patch("ingestion.geocoder.requests.Session.get")
    def test_geocode_outside_india_raises(self, mock_get, mock_sleep):
        mock_get.return_value = self._make_mock_response(
            self._mock_response_outside_india
        )
        with pytest.raises(ValueError, match="outside India"):
            geocode("London")


# ===========================================================================
# Weather client tests
# ===========================================================================

class TestParseHourlyResponse:
    """Tests the internal JSON→DataFrame parser."""

    _sample_payload = {
        "hourly": {
            "time": [
                "2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"
            ],
            "temperature_2m":       [22.5, 23.1, 21.8],
            "relative_humidity_2m": [60.0, 62.0, 65.0],
            "wind_speed_10m":       [10.8, 7.2, 14.4],  # km/h → 3.0, 2.0, 4.0 m/s
            "precipitation":        [0.0, 0.0, 0.2],
            "surface_pressure":     [1013.0, 1012.5, 1012.0],
        }
    }

    def test_parse_returns_dataframe(self):
        df = _parse_hourly_response(self._sample_payload)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_columns_renamed(self):
        df = _parse_hourly_response(self._sample_payload)
        assert "temp_c" in df.columns
        assert "humidity_pct" in df.columns
        assert "wind_speed_ms" in df.columns
        assert "temperature_2m" not in df.columns

    def test_wind_converted_to_ms(self):
        df = _parse_hourly_response(self._sample_payload)
        # 10.8 km/h ÷ 3.6 = 3.0 m/s
        assert abs(df["wind_speed_ms"].iloc[0] - 3.0) < 1e-4

    def test_index_is_utc_datetime(self):
        df = _parse_hourly_response(self._sample_payload)
        assert hasattr(df.index, "tz")
        assert str(df.index.tz) == "UTC"

    def test_missing_hourly_block_raises(self):
        with pytest.raises(ValueError, match="missing 'hourly' block"):
            _parse_hourly_response({"latitude": 18.5, "longitude": 73.8})


# ===========================================================================
# Mock satellite + pipeline smoke test
# ===========================================================================

class TestMockSatellite:
    """Tests the mock GEE fallback without requiring credentials."""

    def test_mock_returns_correct_schema(self):
        from ingestion.gee_client import fetch_satellite_data_mock
        df = fetch_satellite_data_mock(lat=28.6, lon=77.2, lookback_days=2)

        assert isinstance(df, pd.DataFrame)
        expected_cols = {"no2", "so2", "co", "o3", "aod", "lat", "lon"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) >= 48  # at least 2 days × 24h

    def test_mock_values_in_range(self):
        from ingestion.gee_client import fetch_satellite_data_mock
        df = fetch_satellite_data_mock(lat=19.07, lon=72.87, seed=0)

        assert df["no2"].dropna().gt(0).all(), "NO₂ should be positive"
        assert df["aod"].dropna().between(0.0, 1.0).mean() > 0.9, \
            "AOD should mostly be in [0, 1]"


@pytest.mark.skip(reason="Requires GEE credentials and network. Run manually.")
class TestPipelineLive:
    def test_pipeline_delhi(self):
        from ingestion.pipeline import run_ingestion_pipeline
        geo, df = run_ingestion_pipeline("New Delhi", use_mock_satellite=True)
        assert abs(geo.lat - 28.6) < 1.0
        assert not df.empty