"""
forecasting/ensemble.py
-----------------------
Module 4 — LSTM + GRU + XGBoost AQI Forecast Ensemble.

Replaces the old stub that returned random numbers.  This module:

  1. Loads the LSTM and GRU checkpoints (forecasting/checkpoints/).
  2. Loads the XGBoost forecaster artifact (forecasting/checkpoints/
     xgb_forecaster.joblib) trained alongside the neural nets.
  3. Generates a 24-hour autoregressive forecast from each model.
  4. Blends them using validation-tuned weights (default: 0.45 LSTM,
     0.30 GRU, 0.25 XGBoost — overridden by checkpoints/ensemble_weights.json
     if present).

Weights live in JSON, not hard-coded, so retraining can update them
without touching code.

Output format mirrors what api/routes/forecast.py already expects, so the
FastAPI / Streamlit layers don't need any change.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from config.settings import FORECAST_HOURS, LSTM_LOOKBACK_HOURS
from exceptions import CheckpointNotFoundError
from forecasting.dataset import LSTM_FEATURE_COLS
from forecasting.model import (
    AQIForecastGRU,
    AQIForecastLSTM,
    load_checkpoint,
)
from logger import get_logger
from utils import aqi_category

logger = get_logger(__name__)

CKPT_DIR        = Path(__file__).parent / "checkpoints"
XGB_PATH        = CKPT_DIR / "xgb_forecaster.joblib"
WEIGHTS_PATH    = CKPT_DIR / "ensemble_weights.json"

DEFAULT_WEIGHTS = {"lstm": 0.45, "gru": 0.30, "xgb": 0.25}


# ===========================================================================
# Helpers
# ===========================================================================

def load_ensemble_weights() -> dict:
    """
    Return blend weights. Reads checkpoints/ensemble_weights.json if present,
    otherwise falls back to sensible defaults. Always normalised to sum to 1.
    """
    if WEIGHTS_PATH.exists():
        try:
            w = json.loads(WEIGHTS_PATH.read_text())
            # Only keep keys we know about
            w = {k: float(w.get(k, 0.0)) for k in DEFAULT_WEIGHTS}
        except (ValueError, OSError) as e:
            logger.warning("Could not parse {} ({}); using defaults.", WEIGHTS_PATH, e)
            w = dict(DEFAULT_WEIGHTS)
    else:
        w = dict(DEFAULT_WEIGHTS)

    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def _prepare_window(sequence_df: pd.DataFrame, scaler_params: np.ndarray):
    """Apply training-time scaler + return the last 168-hour normalised window."""
    df = sequence_df.copy()
    for col in LSTM_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[LSTM_FEATURE_COLS].ffill(limit=3).bfill(limit=3)
    df = df.fillna(df.median())

    mean = scaler_params[0]
    std  = np.where(scaler_params[1] == 0, 1.0, scaler_params[1])
    norm = (df.values.astype(np.float32) - mean) / std

    if len(norm) < LSTM_LOOKBACK_HOURS:
        raise ValueError(
            f"Need at least {LSTM_LOOKBACK_HOURS} hours of history, got {len(norm)}."
        )
    window = norm[-LSTM_LOOKBACK_HOURS:].copy()
    return df, window, mean, std


def _autoregressive_nn(model, window: np.ndarray, device, hours: int) -> list[float]:
    """Run an LSTM/GRU forecaster autoregressively for `hours` steps."""
    preds_norm = []
    cur = window.copy()
    with torch.no_grad():
        for _ in range(hours):
            x = torch.from_numpy(cur[np.newaxis]).float().to(device)
            p = float(model(x).cpu().numpy().squeeze())
            preds_norm.append(p)
            new = cur[-1].copy()
            new[0] = p   # update AQI; weather held at persistence
            cur = np.vstack([cur[1:], new])
    return preds_norm


def _autoregressive_xgb(xgb_model, window: np.ndarray, hours: int) -> list[float]:
    """
    Run XGBoost autoregressively. The forecaster XGBoost is trained on a
    flattened recent-history vector (last K hours of all features) and
    predicts the NEXT AQI value, normalised in the same scaler space as
    the NN models.
    """
    preds_norm = []
    cur = window.copy()
    # Use last 24 hours of features as the tabular input — matches train.py
    k = min(24, cur.shape[0])
    for _ in range(hours):
        x_flat = cur[-k:].flatten().reshape(1, -1)
        p = float(xgb_model.predict(x_flat).squeeze())
        preds_norm.append(p)
        new = cur[-1].copy()
        new[0] = p
        cur = np.vstack([cur[1:], new])
    return preds_norm


# ===========================================================================
# Public API
# ===========================================================================

def ensemble_forecast_24h(
    sequence_df: pd.DataFrame,
    forecast_hours: int = FORECAST_HOURS,
    weights: Optional[dict] = None,
) -> list[dict]:
    """
    Produce a 24-hour blended AQI forecast from LSTM + GRU + XGBoost.

    Parameters
    ----------
    sequence_df : pd.DataFrame
        Hourly DataFrame with at least LSTM_LOOKBACK_HOURS rows and a
        DatetimeIndex (UTC). Must contain the columns in LSTM_FEATURE_COLS
        (missing ones are filled with 0).
    forecast_hours : int
        Forecast horizon. Default 24.
    weights : dict, optional
        Override blend weights, e.g. {"lstm": 0.5, "gru": 0.3, "xgb": 0.2}.
        Defaults to load_ensemble_weights().

    Returns
    -------
    list[dict] — one dict per forecast hour, matching the schema the
    FastAPI /forecast endpoint already consumes.
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = weights or load_ensemble_weights()

    # --- Load LSTM (mandatory) ----------------------------------------
    lstm_model, scaler_params = load_checkpoint(device, model_type="lstm")
    _, window, mean, std = _prepare_window(sequence_df, scaler_params)

    # --- Try to load GRU & XGB (optional, but warned if missing) ------
    try:
        gru_model, _ = load_checkpoint(device, model_type="gru")
        have_gru = True
    except CheckpointNotFoundError:
        logger.warning("GRU checkpoint missing — falling back to LSTM-only.")
        have_gru = False

    have_xgb = XGB_PATH.exists()
    if have_xgb:
        xgb_model = joblib.load(XGB_PATH)
    else:
        logger.warning("XGBoost forecaster missing — ensemble will skip it.")
        xgb_model = None

    # --- Run each model autoregressively ------------------------------
    lstm_preds = _autoregressive_nn(lstm_model, window, device, forecast_hours)
    gru_preds  = _autoregressive_nn(gru_model,  window, device, forecast_hours) \
                 if have_gru else lstm_preds
    xgb_preds  = _autoregressive_xgb(xgb_model, window, forecast_hours) \
                 if have_xgb else lstm_preds

    # --- Renormalise weights to the models we actually have -----------
    eff_weights = {
        "lstm": weights["lstm"],
        "gru":  weights["gru"] if have_gru else 0.0,
        "xgb":  weights["xgb"] if have_xgb else 0.0,
    }
    total = sum(eff_weights.values()) or 1.0
    eff_weights = {k: v / total for k, v in eff_weights.items()}

    # --- Blend & inverse-transform ------------------------------------
    last_ts = sequence_df.index[-1]
    out = []
    aqi_mean, aqi_std = float(mean[0]), float(std[0])

    for h in range(forecast_hours):
        blended_norm = (
            eff_weights["lstm"] * lstm_preds[h]
            + eff_weights["gru"]  * gru_preds[h]
            + eff_weights["xgb"]  * xgb_preds[h]
        )
        blended_aqi = max(0, round(blended_norm * aqi_std + aqi_mean))
        lstm_aqi    = max(0, round(lstm_preds[h] * aqi_std + aqi_mean))
        gru_aqi     = max(0, round(gru_preds[h]  * aqi_std + aqi_mean))
        xgb_aqi     = max(0, round(xgb_preds[h]  * aqi_std + aqi_mean))

        out.append({
            "forecast_target_time":  last_ts + pd.Timedelta(hours=h + 1),
            "hours_ahead":           h + 1,
            "aqi_forecast":          int(blended_aqi),
            "aqi_category_forecast": aqi_category(blended_aqi),
            "aqi_lstm":              int(lstm_aqi),
            "aqi_gru":               int(gru_aqi),
            "aqi_xgb":               int(xgb_aqi),
            "ensemble_weights":      eff_weights,
        })

    logger.info(
        "Ensemble 24h forecast | weights={} | AQI range [{} – {}]",
        eff_weights,
        min(o["aqi_forecast"] for o in out),
        max(o["aqi_forecast"] for o in out),
    )
    return out