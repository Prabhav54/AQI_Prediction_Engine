"""
forecasting/dataset.py
----------------------
Module 4 — LSTM Dataset Builder

Takes the 168-hour feature sequence from the database and converts
it into (input_sequence, target) pairs that PyTorch can train on.

The core idea is a sliding window:
  - Input X : last 168 hours of AQI + weather features (the "look-back")
  - Target y : the AQI value at the next hour (what we're predicting)

For a 7-day sequence this gives us one training sample per hour.
During inference we take the most recent 168-hour window and predict
T+1 through T+24 autoregressively (feed each prediction back as input).

Why 168 hours?
  - 7 days captures a full weekly cycle (weekday vs weekend patterns)
  - Captures monsoon vs dry day-to-day variation
  - Matches CPCB's own rolling window convention
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import FORECAST_HOURS, LSTM_LOOKBACK_HOURS
from exceptions import SequenceTooShortError
from logger import get_logger

logger = get_logger(__name__)

# Features fed into the LSTM at each timestep
# AQI is first so we can easily split it from weather features
LSTM_FEATURE_COLS = [
    "aqi",              # the primary target variable (also an input)
    "pm25_24h_avg",     # rolling average from DB — already smoothed
    "pm10_24h_avg",
    "no2_24h_avg",
    "so2_24h_avg",
    "co_8h_max",
    "o3_8h_max",
    "temp_c",
    "humidity_pct",
    "wind_speed_ms",
    "precip_mm",
    "pressure_hpa",
    "boundary_layer_m",
]

N_FEATURES = len(LSTM_FEATURE_COLS)


def prepare_sequence(
    df: pd.DataFrame,
    lookback: int = LSTM_LOOKBACK_HOURS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clean, normalise, and structure the DataFrame into arrays
    ready for PyTorch training.

    Parameters
    ----------
    df : pd.DataFrame
        Output of db_client.get_lstm_input_sequence() — hourly rows
        with aqi + weather columns.
    lookback : int
        Input window size (hours). Default: 168.

    Returns
    -------
    X : np.ndarray, shape (n_samples, lookback, n_features)
        Input sequences — each row is a 168-hour window.
    y : np.ndarray, shape (n_samples,)
        Target AQI values — one hour ahead of each window.
    scaler_params : np.ndarray, shape (2, n_features)
        [mean, std] used for normalisation — saved alongside the model
        so inference can apply the same scaling.

    Raises
    ------
    SequenceTooShortError
        If the DataFrame has fewer rows than lookback + 1.
    """
    if len(df) < lookback + 1:
        raise SequenceTooShortError(
            available=len(df),
            required=lookback + 1
        )

    # Only keep columns we actually want
    available = [c for c in LSTM_FEATURE_COLS if c in df.columns]
    missing   = [c for c in LSTM_FEATURE_COLS if c not in df.columns]

    if missing:
        logger.warning(
            "LSTM features not found in sequence data: {} — filling with 0.",
            missing
        )
        for col in missing:
            df[col] = 0.0

    df = df[LSTM_FEATURE_COLS].copy()

    # Forward-fill short gaps, then fill remaining NaNs with column median
    # (NaNs in LSTM inputs cause NaN gradients and silent training failure)
    df = df.ffill(limit=3).bfill(limit=3)
    df = df.fillna(df.median(numeric_only=True)).fillna(0)

    values = df.values.astype(np.float32)

    # Normalise: (x - mean) / std  — per feature, computed on this sequence
    # We save these params so the same scaling applies at inference time
    feature_mean = values.mean(axis=0)
    feature_std  = values.std(axis=0)
    feature_std[feature_std == 0] = 1.0  # avoid division by zero for constant cols

    values_norm  = (values - feature_mean) / feature_std
    scaler_params = np.stack([feature_mean, feature_std])  # shape: (2, n_features)

    # Build sliding windows
    X_list, y_list = [], []

    for i in range(len(values_norm) - lookback):
        X_list.append(values_norm[i : i + lookback])    # window of 168 hours
        y_list.append(values_norm[i + lookback, 0])     # AQI at hour 169 (normalised)

    X = np.stack(X_list)  # (n_samples, lookback, n_features)
    y = np.array(y_list)  # (n_samples,)

    logger.info(
        "Prepared LSTM dataset: X={}, y={}, features={}",
        X.shape, y.shape, N_FEATURES
    )

    return X, y, scaler_params


class AQISequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper around the prepared sequence arrays.
    Passed directly to a DataLoader for batched training.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()  # (n_samples, lookback, n_features)
        self.y = torch.from_numpy(y).float()  # (n_samples,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]