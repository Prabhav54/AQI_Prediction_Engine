"""
forecasting/model.py
--------------------
Module 4 — LSTM Forecasting Model

Architecture: stacked LSTM → dropout → fully connected output

Why LSTM for AQI forecasting?
  - AQI has strong temporal dependencies (today's pollution affects tomorrow's)
  - Weather patterns follow sequences the LSTM can learn
  - 168-hour lookback captures weekly cycles and meteorological persistence

The model predicts one step at a time. For 24-hour forecasting, we
call it autoregressively: predict T+1, feed that prediction back in,
predict T+2, and so on up to T+24.

This file also contains the inference function used by the FastAPI
endpoint — no training code here, just the architecture + predict().
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import FORECAST_HOURS, LSTM_LOOKBACK_HOURS
from exceptions import CheckpointNotFoundError
from forecasting.dataset import LSTM_FEATURE_COLS, N_FEATURES
from logger import get_logger
from utils import aqi_category

logger = get_logger(__name__)

CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "lstm_aqi.pt"


# ================================================================
# Model Architecture
# ================================================================

class AQIForecastLSTM(nn.Module):
    """
    Stacked LSTM for single-step AQI prediction.

    Input  : (batch, seq_len, n_features) — 168 hours of features
    Output : (batch, 1) — predicted normalised AQI at the next hour

    Architecture choices:
    - 2 LSTM layers to capture both short-term and longer-term patterns
    - Hidden size 128 — enough capacity without overfitting on city-level data
    - Dropout 0.2 between layers — regularisation, important since we have
      limited real training data
    - Single linear output head — regression, not classification
    """

    def __init__(
        self,
        n_features:  int   = N_FEATURES,
        hidden_size: int   = 128,
        n_layers:    int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            dropout      = dropout if n_layers > 1 else 0.0,
            batch_first  = True,   # input shape: (batch, seq, features)
        )

        # Small dropout before the output layer adds a bit more regularisation
        self.dropout = nn.Dropout(dropout)

        # Output: predict a single normalised AQI value
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        torch.Tensor, shape (batch, 1)
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # We only care about the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden_size)

        out = self.dropout(last_hidden)
        out = self.fc(out)                  # (batch, 1)

        return out


# ================================================================
# Checkpoint save / load
# ================================================================

def save_checkpoint(
    model: AQIForecastLSTM,
    scaler_params: np.ndarray,
    epoch: int,
    val_loss: float,
) -> None:
    """Save model weights + scaler params so inference can run standalone."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    torch.save({
        "epoch":         epoch,
        "val_loss":      val_loss,
        "model_state":   model.state_dict(),
        "scaler_params": scaler_params,   # shape (2, n_features): [mean, std]
        "n_features":    model.lstm.input_size,
        "hidden_size":   model.hidden_size,
        "n_layers":      model.n_layers,
    }, CHECKPOINT_PATH)

    logger.info(
        "Checkpoint saved → {} (epoch={}, val_loss={:.4f})",
        CHECKPOINT_PATH, epoch, val_loss
    )


def load_checkpoint(device: torch.device) -> tuple[AQIForecastLSTM, np.ndarray]:
    """
    Load model + scaler from the saved checkpoint.

    Returns (model, scaler_params) ready for inference.
    Raises CheckpointNotFoundError if the file doesn't exist.
    """
    if not CHECKPOINT_PATH.exists():
        raise CheckpointNotFoundError(
            f"LSTM checkpoint not found at {CHECKPOINT_PATH}. "
            "Run: python forecasting/train.py"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = AQIForecastLSTM(
        n_features  = checkpoint["n_features"],
        hidden_size = checkpoint["hidden_size"],
        n_layers    = checkpoint["n_layers"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    scaler_params = checkpoint["scaler_params"]

    logger.info(
        "LSTM checkpoint loaded (epoch={}, val_loss={:.4f})",
        checkpoint["epoch"], checkpoint["val_loss"]
    )

    return model, scaler_params


# ================================================================
# 24-hour autoregressive inference
# ================================================================

def forecast_24h(
    sequence_df,                     # pd.DataFrame from db_client
    forecast_hours: int = FORECAST_HOURS,
) -> list[dict]:
    """
    Generate a 24-hour AQI forecast for a location.

    Uses autoregressive prediction: predict T+1, append it to the
    input window, slide the window forward, predict T+2, and so on.
    Weather features for future hours are held constant at their
    last known values (persistence forecast for weather).

    Parameters
    ----------
    sequence_df : pd.DataFrame
        168-hour feature sequence from db_client.get_lstm_input_sequence().
    forecast_hours : int
        Number of hours to forecast ahead. Default 24.

    Returns
    -------
    list[dict]
        One dict per forecast hour:
        {forecast_target_time, aqi_forecast, aqi_category_forecast,
         hours_ahead}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler_params = load_checkpoint(device)

    # Import here to avoid circular dependency
    from forecasting.dataset import prepare_sequence, LSTM_FEATURE_COLS

    # Prepare the input sequence
    X, _, _ = prepare_sequence(sequence_df)

    # We'll use the last window as the seed for autoregression
    # scaler_params from this sequence for inverse-transform
    _, _, scaler_params = prepare_sequence(sequence_df)
    feature_mean = scaler_params[0]
    feature_std  = scaler_params[1]

    # Get the last 168-hour normalised window
    available_cols  = [c for c in LSTM_FEATURE_COLS if c in sequence_df.columns]
    seq_values      = sequence_df[available_cols].values.astype(np.float32)
    seq_norm        = (seq_values - feature_mean) / feature_std
    current_window  = seq_norm[-LSTM_LOOKBACK_HOURS:].copy()  # (168, n_features)

    # The last known timestamp in the sequence
    last_timestamp = sequence_df.index[-1]

    forecasts = []
    model.eval()

    with torch.no_grad():
        for h in range(1, forecast_hours + 1):
            # Input tensor: (1, 168, n_features)
            x_tensor = torch.from_numpy(
                current_window[np.newaxis, :, :]
            ).float().to(device)

            # Predict next normalised AQI
            pred_norm = model(x_tensor).cpu().numpy().squeeze()  # scalar

            # Inverse-transform to get actual AQI value
            aqi_pred = float(pred_norm * feature_std[0] + feature_mean[0])
            aqi_pred = max(0, round(aqi_pred))   # AQI can't be negative

            target_time = last_timestamp + pd.Timedelta(hours=h)

            forecasts.append({
                "forecast_target_time":   target_time,
                "aqi_forecast":           aqi_pred,
                "aqi_category_forecast":  aqi_category(aqi_pred),
                "hours_ahead":            h,
            })

            # Slide the window forward: drop the oldest hour, append the new prediction
            # For the predicted hour, copy weather features from the last known hour
            # (persistence assumption — no weather forecast integration yet)
            new_row = current_window[-1].copy()  # copy last row's weather
            new_row[0] = pred_norm               # update AQI column with our prediction
            current_window = np.vstack([current_window[1:], new_row])

    logger.info(
        "24h forecast generated: AQI range [{} – {}]",
        min(f["aqi_forecast"] for f in forecasts),
        max(f["aqi_forecast"] for f in forecasts),
    )

    return forecasts


# needed for autoregressive loop above
import pandas as pd