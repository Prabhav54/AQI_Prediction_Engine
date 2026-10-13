"""
tests/test_weather_client.py
-----------------------------
Tests for the batched Open-Meteo client (ingestion/weather_client.py).
All network calls are mocked — these tests run fully offline.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ingestion.weather_client import (
    AQ_VARS,
    WEATHER_VARS,
    _parse_hourly_response,
    call_weight,
    fetch_batch,
    split_observed_future,
)

TIMES = ["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T02:00"]


def _payload(vars_: dict, base: float) -> dict:
    return {"hourly": {"time": TIMES, **{k: [base + i for i in range(3)] for k in vars_}}}


def _session(n_locations: int):
    """Mock session: weather call first, then air-quality call, each a list of locations."""
    def get(url, params, timeout):
        resp = MagicMock(status_code=200)
        vars_ = AQ_VARS if "air-quality" in url else WEATHER_VARS
        base = 100.0 if "air-quality" in url else 10.0
        body = [_payload(vars_, base + j) for j in range(n_locations)]
        resp.json.return_value = body if n_locations > 1 else body[0]
        return resp
    s = MagicMock()
    s.get.side_effect = get
    return s


@pytest.fixture
def nodes():
    return pd.DataFrame({"location_hash": ["aaaa0001", "aaaa0002"],
                         "lat": [28.61, 19.08], "lon": [77.21, 72.88]})


class TestParse:
    def test_renames_and_indexes(self):
        df = _parse_hourly_response(_payload(WEATHER_VARS, 1.0), WEATHER_VARS)
        assert list(df.columns) == list(WEATHER_VARS.values())
        assert str(df.index.tz) == "UTC"
        assert df.index.is_monotonic_increasing

    def test_missing_time_axis_raises(self):
        with pytest.raises(ValueError):
            _parse_hourly_response({"latitude": 1.0}, WEATHER_VARS)


class TestFetchBatch:
    def test_multi_location_long_frame(self, nodes):
        df = fetch_batch(nodes, past_days=1, forecast_days=1, session=_session(2))
        assert len(df) == 6
        assert set(df["location_hash"]) == set(nodes["location_hash"])
        for col in list(WEATHER_VARS.values()) + list(AQ_VARS.values()):
            assert col in df.columns

    def test_unit_conversions(self, nodes):
        df = fetch_batch(nodes, past_days=1, session=_session(2))
        first = df[df["location_hash"] == "aaaa0001"].iloc[0]
        assert first["wind_speed_ms"] == pytest.approx(10.0 / 3.6)   # km/h → m/s
        assert first["co"] == pytest.approx(100.0 / 1000.0)          # µg/m³ → mg/m³

    def test_single_location_dict_payload(self, nodes):
        df = fetch_batch(nodes.iloc[:1], past_days=1, session=_session(1))
        assert df["location_hash"].unique().tolist() == ["aaaa0001"]


def test_call_weight_matches_open_meteo_rules():
    assert call_weight(10, 7, 8) == 10              # short range, few vars → 1 per location
    assert call_weight(10, 28, 8) == 20             # 4 weeks → 2 per location
    assert call_weight(1, 14, 20) == 2              # 20 variables → 2


def test_split_observed_future():
    ts = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "x": np.arange(10)})
    obs, fut = split_observed_future(df, now=ts[4] + pd.Timedelta(minutes=30))
    assert obs["timestamp"].max() == ts[4]
    assert fut["timestamp"].min() == ts[5]
