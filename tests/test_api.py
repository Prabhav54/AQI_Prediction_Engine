"""
tests/test_api.py
-----------------
FastAPI read endpoints with the database layer stubbed out.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.routes.forecast as fr
from api.main import app

NODE = {"location_hash": "abcd1234", "location_name": "Delhi", "state": "Delhi",
        "lat": 28.61, "lon": 77.21, "kind": "city", "distance_km": 1.2}
NOW = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(fr, "nearest_node", lambda lat, lon: NODE)
    monkeypatch.setattr(fr, "latest_aqi", lambda h: {
        **NODE, "timestamp": NOW, "aqi": 180, "aqi_category": "Moderate",
        "prominent_pollutant": "PM2.5", "pm25_24h_avg": 84.0, "o3_8h_avg": float("nan"),
    })
    monkeypatch.setattr(fr, "latest_forecast", lambda h: pd.DataFrame([{
        "forecast_generated_at": NOW, "forecast_target_time": NOW + pd.Timedelta(hours=k),
        "hours_ahead": k, "aqi": 170 + k, "aqi_lower": 150, "aqi_upper": 200,
        "aqi_category": "Moderate", "prominent_pollutant": "PM2.5", "pm25": 80.0,
        "model_version": "test",
    } for k in range(1, 25)]))
    return TestClient(app)


def test_aqi(client):
    r = client.get("/aqi", params={"lat": 28.6, "lon": 77.2})
    assert r.status_code == 200
    body = r.json()
    assert body["aqi"] == 180 and body["node"]["location_name"] == "Delhi"
    assert body["pollutants"]["o3_8h_avg"] is None          # NaN → null


def test_forecast(client):
    r = client.get("/forecast", params={"lat": 28.6, "lon": 77.2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["hourly"]) == 24
    assert body["aqi_max"] == 194 and body["dominant_category"] == "Moderate"


def test_rejects_outside_india(client):
    assert client.get("/aqi", params={"lat": 51.5, "lon": -0.1}).status_code == 422


def test_ingest_validates_resolution(client):
    assert client.post("/ingest/pan-india", params={"resolution": 0.1}).status_code == 400
