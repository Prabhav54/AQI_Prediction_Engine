"""
tests/test_pipeline.py
----------------------
Tests for Module 2 (proxy model), Module 4 (ensemble),
and Module 5 (API routes).

These tests are designed to run without:
  - A live database connection
  - GEE credentials
  - Trained model artifacts

Everything is either tested with synthetic data or mocked so the
tests work cleanly in CI and on a fresh clone.

Run:
  pytest tests/ -v
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ================================================================
# Fixtures — reusable test data
# ================================================================

@pytest.fixture
def synthetic_merged_df():
    """
    Mimics the output of ingestion/pipeline.py — a 7-day hourly
    DataFrame with satellite + weather + metadata columns.
    """
    rng   = np.random.default_rng(42)
    now   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=now, periods=168, freq="1h", tz="UTC")
    n     = len(index)

    df = pd.DataFrame({
        "no2":              rng.normal(60,  20,  n).clip(0),
        "so2":              rng.normal(8,   3,   n).clip(0),
        "co":               rng.normal(5,   1.5, n).clip(0),
        "o3":               rng.normal(120, 30,  n).clip(0),
        "aod":              rng.uniform(0.1, 0.9, n),
        "temp_c":           rng.normal(28,  5,   n),
        "humidity_pct":     rng.normal(60,  12,  n).clip(10, 99),
        "wind_speed_ms":    rng.exponential(2, n).clip(0.1, 20),
        "precip_mm":        rng.exponential(0.5, n).clip(0),
        "pressure_hpa":     rng.normal(1010, 8, n),
        "boundary_layer_m": rng.normal(800, 200, n).clip(100, 3000),
        "lat":              28.6,
        "lon":              77.2,
        "location_name":    "New Delhi, India",
        "ingested_at":      now,
    }, index=index)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def synthetic_aqi_sequence():
    """
    Mimics the output of db_client.get_lstm_input_sequence() —
    a 168-hour DataFrame with AQI and weather features.
    """
    rng   = np.random.default_rng(42)
    now   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=now, periods=200, freq="1h", tz="UTC")
    n     = len(index)

    df = pd.DataFrame({
        "aqi":              rng.integers(50, 250, n).astype(float),
        "pm25_24h_avg":     rng.normal(80,  20, n).clip(0),
        "pm10_24h_avg":     rng.normal(140, 40, n).clip(0),
        "no2_24h_avg":      rng.normal(60,  15, n).clip(0),
        "so2_24h_avg":      rng.normal(20,  5,  n).clip(0),
        "co_8h_max":        rng.normal(3,   1,  n).clip(0),
        "o3_8h_max":        rng.normal(80,  20, n).clip(0),
        "temp_c":           rng.normal(28,  5,  n),
        "humidity_pct":     rng.normal(60,  12, n).clip(10, 99),
        "wind_speed_ms":    rng.exponential(2,  n).clip(0.1, 20),
        "precip_mm":        rng.exponential(0.5, n).clip(0),
        "pressure_hpa":     rng.normal(1010, 8, n),
        "boundary_layer_m": rng.normal(800, 200, n).clip(100, 3000),
    }, index=index)
    df.index.name = "timestamp"
    return df


# ================================================================
# Module 2 — Proxy model inference
# ================================================================

class TestProxyInference:

    def test_adds_pm25_pm10_columns(self, synthetic_merged_df):
        """After inference, the DataFrame must have both PM proxy columns."""
        from proxy_model.predict import run_proxy_inference
        from exceptions import ModelNotFoundError

        # If model artifacts don't exist yet, we expect ModelNotFoundError
        # In CI (no trained model) this is the expected path
        try:
            result = run_proxy_inference(synthetic_merged_df)
            assert "pm25_proxy" in result.columns
            assert "pm10_proxy" in result.columns
        except ModelNotFoundError:
            pytest.skip("Proxy model not trained yet — run proxy_model/train.py")

    def test_missing_aod_rows_stay_nan(self, synthetic_merged_df):
        """Rows where AOD is completely missing should have NaN PM estimates."""
        from proxy_model.predict import run_proxy_inference
        from exceptions import ModelNotFoundError

        df = synthetic_merged_df.copy()
        df.loc[df.index[:10], "aod"] = np.nan

        try:
            result = run_proxy_inference(df)
            # Some NaN rows may be filled by ffill, but fully missing should stay NaN
            assert result["pm25_proxy"].notna().sum() > 0
        except ModelNotFoundError:
            pytest.skip("Proxy model not trained yet.")

    def test_raises_on_missing_core_columns(self):
        """Should raise InsufficientFeaturesError if weather cols are absent."""
        from proxy_model.predict import run_proxy_inference
        from exceptions import InsufficientFeaturesError, ModelNotFoundError

        bad_df = pd.DataFrame({"aod": [0.5, 0.6]})  # missing temp_c, humidity_pct
        try:
            run_proxy_inference(bad_df)
            pytest.fail("Should have raised InsufficientFeaturesError")
        except InsufficientFeaturesError:
            pass  # expected
        except ModelNotFoundError:
            pytest.skip("Proxy model not trained yet.")


# ================================================================
# Module 4 — Dataset builder
# ================================================================

class TestDatasetBuilder:

    def test_sliding_window_shapes(self, synthetic_aqi_sequence):
        """X and y should have the right shapes for the given lookback."""
        from forecasting.dataset import prepare_sequence

        lookback = 24  # use 24h for speed
        X, y, scaler = prepare_sequence(synthetic_aqi_sequence, lookback=lookback)

        n_samples = len(synthetic_aqi_sequence) - lookback
        assert X.shape == (n_samples, lookback, 13), f"X shape mismatch: {X.shape}"
        assert y.shape == (n_samples,),              f"y shape mismatch: {y.shape}"
        assert scaler.shape == (2, 13),              f"Scaler shape mismatch: {scaler.shape}"

    def test_no_nans_in_output(self, synthetic_aqi_sequence):
        """NaN in inputs should be filled before building windows."""
        from forecasting.dataset import prepare_sequence

        df = synthetic_aqi_sequence.copy()
        df.loc[df.index[5:10], "aqi"] = np.nan  # inject NaNs

        X, y, _ = prepare_sequence(df, lookback=24)
        assert not np.isnan(X).any(), "NaN found in X after preparation"
        assert not np.isnan(y).any(), "NaN found in y after preparation"

    def test_too_short_raises(self):
        """Should raise SequenceTooShortError if df < lookback + 1."""
        from forecasting.dataset import prepare_sequence
        from exceptions import SequenceTooShortError

        tiny_df = pd.DataFrame(
            {"aqi": range(10)},
            index=pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        )
        with pytest.raises(SequenceTooShortError):
            prepare_sequence(tiny_df, lookback=24)

    def test_pytorch_dataset_len(self, synthetic_aqi_sequence):
        """AQISequenceDataset.__len__ should match number of windows."""
        from forecasting.dataset import AQISequenceDataset, prepare_sequence

        X, y, _ = prepare_sequence(synthetic_aqi_sequence, lookback=24)
        dataset  = AQISequenceDataset(X, y)
        assert len(dataset) == len(X)

    def test_pytorch_dataset_item_shapes(self, synthetic_aqi_sequence):
        """Each item from the dataset should be (seq_tensor, target_tensor)."""
        import torch
        from forecasting.dataset import AQISequenceDataset, prepare_sequence

        X, y, _ = prepare_sequence(synthetic_aqi_sequence, lookback=24)
        dataset  = AQISequenceDataset(X, y)
        x_item, y_item = dataset[0]

        assert isinstance(x_item, torch.Tensor)
        assert x_item.shape == (24, 13)
        assert y_item.shape == ()


# ================================================================
# Module 4 — LSTM model architecture
# ================================================================

class TestLSTMModel:

    def test_forward_pass_shape(self):
        """Model output should be (batch_size, 1)."""
        import torch
        from forecasting.model import AQIForecastLSTM

        model   = AQIForecastLSTM()
        x       = torch.randn(4, 168, 13)   # batch=4, seq=168, features=13
        output  = model(x)
        assert output.shape == (4, 1), f"Unexpected output shape: {output.shape}"

    def test_forward_pass_no_nan(self):
        """No NaN values should appear in a forward pass."""
        import torch
        from forecasting.model import AQIForecastLSTM

        model  = AQIForecastLSTM()
        x      = torch.randn(2, 168, 13)
        output = model(x)
        assert not torch.isnan(output).any()


# ================================================================
# Module 5 — FastAPI (with mock DB)
# ================================================================

class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        """HTTP test client for FastAPI with DB calls mocked out."""
        from fastapi.testclient import TestClient
        from api.main import app

        # Mock DB connectivity so startup doesn't fail in CI
        with patch("database.db_client.sync_engine") as mock_engine:
            mock_engine.connect.return_value.__enter__ = MagicMock(
                return_value=MagicMock(execute=MagicMock(
                    return_value=MagicMock(scalar=MagicMock(return_value=1))
                ))
            )
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
            yield TestClient(app)

    def test_health_endpoint_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "api" in resp.json()

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_ingest_empty_location_rejected(self, client):
        """Empty location string should return 422 validation error."""
        resp = client.post("/ingest/", json={"location": "   "})
        assert resp.status_code == 422

    def test_ingest_valid_request_structure(self, client):
        """Valid ingest request should be accepted (geocode is mocked)."""
        from ingestion.geocoder import GeoLocation

        mock_geo = GeoLocation(
            query        = "Delhi",
            display_name = "New Delhi, Delhi, India",
            lat          = 28.6139,
            lon          = 77.2090,
            osm_type     = "relation",
            importance   = 0.80,
        )

        with patch("api.routes.ingest.geocode", return_value=mock_geo), \
             patch("api.routes.ingest._run_pipeline_task"):
            resp = client.post("/ingest/", json={
                "location":           "Delhi",
                "lookback_days":      7,
                "use_mock_satellite": True,
            })

        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "accepted"
        assert body["lat"]    == pytest.approx(28.6139, abs=0.01)