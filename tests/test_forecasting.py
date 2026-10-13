"""
tests/test_forecasting.py
-------------------------
Shape / leakage / plumbing tests for the global forecaster on synthetic data.
No trained checkpoints or network needed.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from config.settings import POLLUTANTS
from forecasting.dataset import (
    FUTURE_FEATURES,
    HORIZON,
    LOOKBACK,
    PAST_FEATURES,
    AQISequenceDataset,
    build_store,
    chronological_splits,
    fit_spec,
    lgbm_feature_names,
    lgbm_features,
    normalise_store,
    sample_index,
)
from forecasting.ensemble import blend, deltas_to_conc
from forecasting.model import AQIForecastGRU, AQIForecastLSTM

RAW = ["temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg", "precip_mm", "pressure_hpa",
       "boundary_layer_m", "radiation_wm2", "pm25", "pm10", "no2", "so2", "co", "o3", "dust", "aod"]


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    hours = 24 * 30
    ts = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    frames = []
    for h, (lat, lon) in {"n0000001": (28.6, 77.2), "n0000002": (19.1, 72.9)}.items():
        df = pd.DataFrame(rng.gamma(3, 10, (hours, len(RAW))), columns=RAW)
        df["wind_dir_deg"] = rng.uniform(0, 360, hours)
        df["timestamp"], df["location_hash"] = ts, h
        frames.append(df)
    frame = pd.concat(frames, ignore_index=True)
    frame.loc[100:103, "pm25"] = np.nan          # a short gap → ffilled
    frame.loc[400:420, "temp_c"] = np.nan        # a long gap → windows touching it invalid
    nodes = pd.DataFrame({"location_hash": ["n0000001", "n0000002"], "lat": [28.6, 19.1], "lon": [77.2, 72.9]})
    return nodes, frame


def test_store_and_validity(synthetic):
    nodes, frame = synthetic
    store = build_store(frame, nodes)
    assert len(store) == 2
    s = store[0]
    assert s.past.shape[1] == len(PAST_FEATURES) and s.future.shape[1] == len(FUTURE_FEATURES)
    assert not s.valid[:LOOKBACK - 1].any()            # need a full lookback
    assert not s.valid[-HORIZON:].any()                # need a full horizon of targets
    assert not s.valid[420]                            # long gap inside the window


def test_splits_do_not_overlap(synthetic):
    nodes, frame = synthetic
    store = build_store(frame, nodes)
    (_, tr_end), (va0, va1), (te0, _) = chronological_splits(store, val_days=4, test_days=4)
    tr = sample_index(store, 1, None, tr_end)
    va = sample_index(store, 1, va0, va1)
    last_train_target = max(store[n].times[t] for n, t in tr) + pd.Timedelta(hours=HORIZON)
    first_val_issue = min(store[n].times[t] for n, t in va)
    assert last_train_target <= first_val_issue        # no target leakage into validation
    assert te0 > va1


def test_dataset_and_models(synthetic):
    nodes, frame = synthetic
    store = build_store(frame, nodes)
    idx = sample_index(store, 12)
    spec = fit_spec(store, idx)
    norm = normalise_store(store, spec)
    past, fut, y = AQISequenceDataset(norm, idx, spec)[0]
    assert past.shape == (LOOKBACK, len(PAST_FEATURES))
    assert fut.shape == (HORIZON, len(FUTURE_FEATURES))
    assert y.shape == (HORIZON, len(POLLUTANTS))
    for cls in (AQIForecastLSTM, AQIForecastGRU):
        out = cls()(past[None], fut[None])
        assert out.shape == (1, HORIZON, len(POLLUTANTS))
        assert torch.isfinite(out).all()


def test_lgbm_features_align_with_names(synthetic):
    nodes, frame = synthetic
    store = build_store(frame, nodes)
    idx = sample_index(store, 24)[:5]
    X = lgbm_features(store, idx, np.full(len(idx), 7))
    assert X.shape == (5, len(lgbm_feature_names()))
    assert X[:, -1].tolist() == [7] * 5                 # horizon column


def test_blend_and_conversion():
    a = np.ones((2, HORIZON, 6))
    preds = {"lstm": a * 0.0, "gru": a * 1.0, "lgbm": a * 2.0}
    w = {p: {"lstm": 0.5, "gru": 0.5, "lgbm": 0.0} for p in POLLUTANTS}
    assert np.allclose(blend(preds, w), 0.5)
    # Missing member → weights renormalised over the members present
    assert np.allclose(blend({"lgbm": a * 2.0}, w), 2.0)
    conc = deltas_to_conc(np.log1p(np.full((2, 6), 9.0)), np.zeros((2, HORIZON, 6)))
    assert np.allclose(conc, 9.0)
