"""
forecasting/model.py
--------------------
Module 4 — LSTM + GRU Forecasting Models

This file defines BOTH neural sequence models used in the ensemble:
  • AQIForecastLSTM — stacked LSTM (captures long-term dependencies)
  • AQIForecastGRU  — stacked GRU  (lighter, often better on noisy data)

Both share the same input shape and same checkpoint protocol so the
ensemble layer (forecasting/ensemble.py) can call either interchangeably.

Why both?
  LSTMs and GRUs make different mistakes on the same input. Averaging
  their predictions (then blending with XGBoost) reduces variance and
  almost always beats either model alone.

Inference is autoregressive: predict T+1, append the prediction to the
input window, slide the window, predict T+2, … up to T+24.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
LSTM_CKPT_PATH  = CHECKPOINT_DIR / "lstm_aqi.pt"
GRU_CKPT_PATH   = CHECKPOINT_DIR / "gru_aqi.pt"


# ===========================================================================
# Architectures
# ===========================================================================

class _RecurrentForecaster(nn.Module):
    """Shared scaffolding for LSTM and GRU forecasters."""

    def __init__(
        self,
        rnn_cls,
        n_features:  int   = N_FEATURES,
        hidden_size: int   = 128,
        n_layers:    int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.rnn = rnn_cls(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout if n_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)          # (B, T, H)
        last   = out[:, -1, :]        # (B, H)
        return self.fc(self.dropout(last))   # (B, 1)


class AQIForecastLSTM(_RecurrentForecaster):
    """Stacked LSTM forecaster."""
    def __init__(self, **kwargs):
        super().__init__(nn.LSTM, **kwargs)
        # alias so save_checkpoint can read `.input_size` regardless of cell type
        self.lstm = self.rnn


class AQIForecastGRU(_RecurrentForecaster):
    """Stacked GRU forecaster — lighter than LSTM, faster to train."""
    def __init__(self, **kwargs):
        super().__init__(nn.GRU, **kwargs)
        self.gru = self.rnn


# ===========================================================================
# Checkpoint save / load
# ===========================================================================

def _ckpt_path_for(model: nn.Module) -> Path:
    return GRU_CKPT_PATH if isinstance(model, AQIForecastGRU) else LSTM_CKPT_PATH


def save_checkpoint(
    model: _RecurrentForecaster,
    scaler_params: np.ndarray,
    epoch: int,
    val_loss: float,
    path: Path | None = None,
) -> None:
    """Persist weights + scaler so inference can run standalone."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    target = path or _ckpt_path_for(model)

    torch.save({
        "epoch":         epoch,
        "val_loss":      val_loss,
        "model_state":   model.state_dict(),
        "scaler_params": scaler_params,            # shape (2, n_features)
        "n_features":    model.rnn.input_size,
        "hidden_size":   model.hidden_size,
        "n_layers":      model.n_layers,
        "model_class":   model.__class__.__name__,
    }, target)

    logger.info(
        "Checkpoint saved → {} (epoch={}, val_loss={:.4f})",
        target, epoch, val_loss,
    )


def load_checkpoint(
    device: torch.device,
    model_type: str = "lstm",
) -> tuple[_RecurrentForecaster, np.ndarray]:
    """
    Load a trained model + its training-time scaler.

    Parameters
    ----------
    model_type : 'lstm' | 'gru'
    """
    if model_type.lower() == "gru":
        path, ctor = GRU_CKPT_PATH, AQIForecastGRU
    else:
        path, ctor = LSTM_CKPT_PATH, AQIForecastLSTM

    if not path.exists():
        raise CheckpointNotFoundError(
            f"{model_type.upper()} checkpoint not found at {path}. "
            f"Run: python forecasting/train.py --model {model_type}"
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = ctor(
        n_features  = ckpt["n_features"],
        hidden_size = ckpt["hidden_size"],
        n_layers    = ckpt["n_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    logger.info(
        "{} checkpoint loaded (epoch={}, val_loss={:.4f})",
        model_type.upper(), ckpt["epoch"], ckpt["val_loss"],
    )
    return model, ckpt["scaler_params"]


# ===========================================================================
# 24-hour autoregressive inference (single model)
# ===========================================================================

def forecast_24h_single(
    sequence_df: pd.DataFrame,
    model_type: str = "lstm",
    forecast_hours: int = FORECAST_HOURS,
) -> list[dict]:
    """
    Generate a 24-hour AQI forecast using a SINGLE neural model
    (LSTM or GRU). Use forecasting/ensemble.py for the blended forecast.

    Critical fix vs. previous version
    ---------------------------------
    The scaler params now come from the SAVED CHECKPOINT (training-time
    statistics), not recomputed from the inference sequence. Re-fitting
    the scaler at inference time silently corrupts predictions whenever
    the recent window's distribution drifts from the training set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler_params = load_checkpoint(device, model_type=model_type)

    feature_mean = scaler_params[0]
    feature_std  = scaler_params[1]
    # Guard against degenerate std (constant column at train time)
    feature_std  = np.where(feature_std == 0, 1.0, feature_std)

    # Make sure every expected feature column is present
    df = sequence_df.copy()
    for col in LSTM_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[LSTM_FEATURE_COLS].ffill(limit=3).bfill(limit=3)
    df = df.fillna(df.median())

    seq_values = df.values.astype(np.float32)
    seq_norm   = (seq_values - feature_mean) / feature_std

    if len(seq_norm) < LSTM_LOOKBACK_HOURS:
        raise ValueError(
            f"Need at least {LSTM_LOOKBACK_HOURS} hours of history, "
            f"got {len(seq_norm)}."
        )

    current_window = seq_norm[-LSTM_LOOKBACK_HOURS:].copy()   # (168, F)
    last_timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) \
        else sequence_df.index[-1]

    forecasts = []
    with torch.no_grad():
        for h in range(1, forecast_hours + 1):
            x = torch.from_numpy(current_window[np.newaxis]).float().to(device)
            pred_norm = float(model(x).cpu().numpy().squeeze())

            # Inverse-transform: column 0 is AQI
            aqi_pred = float(pred_norm * feature_std[0] + feature_mean[0])
            aqi_pred = max(0, round(aqi_pred))

            target_time = last_timestamp + pd.Timedelta(hours=h)
            forecasts.append({
                "forecast_target_time":  target_time,
                "aqi_forecast":          aqi_pred,
                "aqi_category_forecast": aqi_category(aqi_pred),
                "hours_ahead":           h,
                "model":                 model_type.upper(),
            })

            # Slide window: keep weather columns as persistence, update AQI
            new_row    = current_window[-1].copy()
            new_row[0] = pred_norm
            current_window = np.vstack([current_window[1:], new_row])

    logger.info(
        "{} 24h forecast: AQI range [{} – {}]",
        model_type.upper(),
        min(f["aqi_forecast"] for f in forecasts),
        max(f["aqi_forecast"] for f in forecasts),
    )
    return forecasts


# Backwards-compat alias so old imports keep working
def forecast_24h(sequence_df, forecast_hours: int = FORECAST_HOURS) -> list[dict]:
    """Backwards-compatible wrapper — runs the LSTM-only forecast."""
    return forecast_24h_single(sequence_df, "lstm", forecast_hours)