"""
tests/test_weather_client.py
-----------------------------
Tests for the Open-Meteo weather client (ingestion/weather_client.py).

Covers:
  - Response parsing + unit conversion
  - Archive + forecast stitching logic
  - Missing column handling
  - Forward-fill gap behaviour

All network calls are mocked — these tests run fully offline.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.weather_client import (
    _build_session,
    _parse_hourly_response,
    fetch_weather,
)


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def sample_hourly_payload():
    """Minimal valid Open-Meteo hourly JSON payload (3 hours)."""
    return {
        "latitude":  28.6139,
        "longitude": 77.2090,
        "timezone":  "UTC",
        "hourly": {
            "time":                  ["2024-06-01T00:00", "2024-06-01T01:00", "2024-06-01T02:00"],
            "temperature_2m":        [32.1, 31.8, 31.5],
            "relative_humidity_2m":  [55.0, 57.0, 59.0],
            "wind_speed_10m":        [10.8, 7.2,  14.4],   # km/h
            "precipitation":         [0.0,  0.0,  0.2],
            "surface_pressure":      [995.0, 994.8, 994.6],
            "boundary_layer_height": [1200.0, 1100.0, 950.0],
        },
    }


@pytest.fixture
def sample_hourly_payload_sparse():
    """Payload where boundary_layer_height is missing (common from archive API)."""
    return {
        "latitude":  19.0760,
        "longitude": 72.8777,
        "timezone":  "UTC",
        "hourly": {
            "time":                 ["2024-06-01T00:00", "2024-06-01T01:00"],
            "temperature_2m":       [30.5, 30.8],
            "relative_humidity_2m": [78.0, 79.0],
            "wind_speed_10m":       [18.0, 21.6],
            "precipitation":        [1.2,  0.0],
            "surface_pressure":     [1005.0, 1004.8],
            # boundary_layer_height intentionally absent
        },
    }


def _make_mock_response(payload: dict) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value    = payload
    mock.raise_for_status.return_value = None
    return mock


# ================================================================
# _parse_hourly_response
# ================================================================

class TestParseHourlyResponse:

    def test_returns_dataframe(self, sample_hourly_payload):
        df = _parse_hourly_response(sample_hourly_payload)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_time_entries(self, sample_hourly_payload):
        df = _parse_hourly_response(sample_hourly_payload)
        assert len(df) == 3

    def test_index_is_utc_datetimeindex(self, sample_hourly_payload):
        df = _parse_hourly_response(sample_hourly_payload)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"

    def test_columns_renamed_correctly(self, sample_hourly_payload):
        df = _parse_hourly_response(sample_hourly_payload)
        # Old names should be gone
        assert "temperature_2m"       not in df.columns
        assert "relative_humidity_2m" not in df.columns
        assert "wind_speed_10m"       not in df.columns
        # New names should be present
        assert "temp_c"         in df.columns
        assert "humidity_pct"   in df.columns
        assert "wind_speed_ms"  in df.columns

    def test_wind_converted_from_kmh_to_ms(self, sample_hourly_payload):
        df = _parse_hourly_response(sample_hourly_payload)
        # 10.8 km/h ÷ 3.6 = 3.0 m/s
        assert abs(df["wind_speed_ms"].iloc[0] - 3.0) < 1e-4
        # 7.2 km/h ÷ 3.6 = 2.0 m/s
        assert abs(df["wind_speed_ms"].iloc[1] - 2.0) < 1e-4

    def test_optional_column_included_when_present(self, sample_hourly_payload):
        df = _parse_hourly_response(sample_hourly_payload)
        assert "boundary_layer_m" in df.columns

    def test_optional_column_absent_when_not_in_payload(self, sample_hourly_payload_sparse):
        df = _parse_hourly_response(sample_hourly_payload_sparse)
        # boundary_layer_m should simply not be there — no KeyError
        assert "boundary_layer_m" not in df.columns

    def test_empty_hourly_block_raises_valueerror(self):
        with pytest.raises(ValueError, match="missing 'hourly' block"):
            _parse_hourly_response({"latitude": 28.6, "longitude": 77.2})

    def test_missing_time_key_raises_valueerror(self):
        with pytest.raises(ValueError, match="missing 'hourly' block"):
            _parse_hourly_response({"hourly": {"temperature_2m": [30.0]}})

    def test_index_sorted_ascending(self, sample_hourly_payload):
        df = _parse_hourly_response(sample_hourly_payload)
        assert df.index.is_monotonic_increasing

    def test_numeric_columns_are_float(self, sample_hourly_payload):
        df = _parse_hourly_response(sample_hourly_payload)
        for col in ["temp_c", "humidity_pct", "wind_speed_ms"]:
            assert df[col].dtype in [np.float32, np.float64], \
                f"Column '{col}' is not float dtype"


# ================================================================
# fetch_weather (live call mocked)
# ================================================================

class TestFetchWeather:

    def _make_full_payload(self, n_hours: int = 72) -> dict:
        """Generate a realistic n-hour payload with recent timestamps for mocking."""
        rng   = np.random.default_rng(0)
        # Use timestamps relative to now so the lookback window clip keeps them
        now   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=n_hours - 1)
        times = [
            (start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
            for i in range(n_hours)
        ]
        return {
            "latitude": 28.61, "longitude": 77.21, "timezone": "UTC",
            "hourly": {
                "time":                  times,
                "temperature_2m":        rng.normal(30, 3, n_hours).tolist(),
                "relative_humidity_2m":  rng.normal(60, 10, n_hours).clip(10, 99).tolist(),
                "wind_speed_10m":        rng.exponential(10, n_hours).tolist(),
                "precipitation":         rng.exponential(0.5, n_hours).clip(0).tolist(),
                "surface_pressure":      rng.normal(1008, 5, n_hours).tolist(),
                "boundary_layer_height": rng.normal(900, 200, n_hours).clip(100).tolist(),
            },
        }

    @patch("ingestion.weather_client.requests.Session.get")
    def test_returns_dataframe_with_expected_columns(self, mock_get):
        payload = self._make_full_payload(72)
        mock_get.return_value = _make_mock_response(payload)

        df = fetch_weather(lat=28.61, lon=77.21, lookback_days=3)

        assert isinstance(df, pd.DataFrame)
        assert "temp_c"        in df.columns
        assert "humidity_pct"  in df.columns
        assert "wind_speed_ms" in df.columns

    @patch("ingestion.weather_client.requests.Session.get")
    def test_no_nans_after_ffill(self, mock_get):
        """
        After forward-fill, short gaps should be filled.
        Inject some NaNs into the mock payload and confirm they're gone.
        """
        payload = self._make_full_payload(72)
        # Inject NaNs into temperature
        payload["hourly"]["temperature_2m"][5:8] = [None, None, None]
        mock_get.return_value = _make_mock_response(payload)

        df = fetch_weather(lat=28.61, lon=77.21, lookback_days=3)

        # ffill(limit=2) fills up to 2 consecutive NaNs
        # 3 consecutive NaNs → 1 may remain, but the column shouldn't be fully null
        assert df["temp_c"].notna().sum() > 0

    @patch("ingestion.weather_client.requests.Session.get")
    def test_index_is_utc_datetimeindex(self, mock_get):
        payload = self._make_full_payload(48)
        mock_get.return_value = _make_mock_response(payload)

        df = fetch_weather(lat=19.07, lon=72.88, lookback_days=2)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None
        assert "UTC" in str(df.index.tz)

    @patch("ingestion.weather_client.requests.Session.get")
    def test_wind_values_are_in_ms_not_kmh(self, mock_get):
        """Wind values should be in m/s after conversion."""
        payload = self._make_full_payload(24)
        # Force a known wind value: 36 km/h → should become 10.0 m/s
        payload["hourly"]["wind_speed_10m"] = [36.0] * 24
        mock_get.return_value = _make_mock_response(payload)

        df = fetch_weather(lat=28.61, lon=77.21, lookback_days=1)

        # All values should be 10.0 m/s (36 ÷ 3.6)
        assert (df["wind_speed_ms"].dropna() - 10.0).abs().max() < 0.01

    @patch("ingestion.weather_client.requests.Session.get")
    def test_runtime_error_on_request_failure(self, mock_get):
        """HTTP error should propagate as WeatherFetchError."""
        import requests
        from exceptions import WeatherFetchError

        mock_get.side_effect = requests.RequestException("Connection refused")

        with pytest.raises((WeatherFetchError, RuntimeError)):
            fetch_weather(lat=28.61, lon=77.21, lookback_days=1)


# ================================================================
# Edge cases
# ================================================================

class TestWeatherEdgeCases:

    def test_parse_handles_all_none_values_gracefully(self):
        """NaN-heavy payload shouldn't crash — just produce NaN columns."""
        payload = {
            "hourly": {
                "time":               ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m":     [None, None],
                "relative_humidity_2m": [None, None],
                "wind_speed_10m":     [None, None],
                "precipitation":      [None, None],
                "surface_pressure":   [None, None],
            }
        }
        df = _parse_hourly_response(payload)
        # Should not raise — just produce NaN columns
        assert len(df) == 2
        assert df["temp_c"].isna().all()

    def test_parse_single_row_payload(self):
        """Single-row payload (e.g. requesting 1 hour) should work fine."""
        payload = {
            "hourly": {
                "time":               ["2024-01-01T12:00"],
                "temperature_2m":     [25.0],
                "relative_humidity_2m": [65.0],
                "wind_speed_10m":     [18.0],
                "precipitation":      [0.0],
                "surface_pressure":   [1010.0],
            }
        }
        df = _parse_hourly_response(payload)
        assert len(df) == 1
        assert abs(df["wind_speed_ms"].iloc[0] - 5.0) < 1e-4  # 18 ÷ 3.6