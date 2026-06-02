"""
proxy_model/predict.py
----------------------
Module 2 — PM2.5 / PM10 Proxy Model Inference

This is the inference side of the proxy model. After the ingestion
pipeline (Module 1) pulls satellite + weather data, this script
takes that merged DataFrame and appends estimated PM2.5 and PM10
columns to it — ready for the database write in Module 3.

The flow is simple:
    merged_df (from pipeline.py)
        → fill missing AOD/weather features
        → scale features using the saved scaler
        → run XGBoost prediction
        → attach pm25_proxy + pm10_proxy columns back to the DataFrame

If the model artifacts don't exist yet, it tells you to run train.py first.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exceptions import InsufficientFeaturesError, ModelNotFoundError
from logger import get_logger

logger = get_logger(__name__)

# Artifact paths — must match what train.py saved
ARTIFACTS_DIR   = Path(__file__).parent / "artifacts"
PM25_MODEL_PATH = ARTIFACTS_DIR / "xgb_pm25_proxy.joblib"
PM10_MODEL_PATH = ARTIFACTS_DIR / "xgb_pm10_proxy.joblib"
SCALER_PATH     = ARTIFACTS_DIR / "feature_scaler.joblib"

# These must exactly match the column list in train.py
FEATURE_COLS = [
    "aod",
    "temp_c",
    "humidity_pct",
    "wind_speed_ms",
    "pressure_hpa",
    "boundary_layer_m",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def _load_artifacts() -> tuple:
    """
    Load the trained XGBoost models and scaler from disk.
    Raises ModelNotFoundError with a helpful message if any artifact
    is missing — usually means train.py hasn't been run yet.
    """
    for path in [PM25_MODEL_PATH, PM10_MODEL_PATH, SCALER_PATH]:
        if not path.exists():
            raise ModelNotFoundError(str(path))

    pm25_model = joblib.load(PM25_MODEL_PATH)
    pm10_model = joblib.load(PM10_MODEL_PATH)
    scaler     = joblib.load(SCALER_PATH)

    logger.info("Proxy model artifacts loaded from: {}", ARTIFACTS_DIR)
    return pm25_model, pm10_model, scaler


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic hour and month encodings to the DataFrame.
    The model was trained with these features so we need them at
    inference time too. Uses the DatetimeIndex of the DataFrame.
    """
    df = df.copy()
    hours  = df.index.hour
    months = df.index.month

    df["hour_sin"]  = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hours / 24)
    df["month_sin"] = np.sin(2 * np.pi * months / 12)
    df["month_cos"] = np.cos(2 * np.pi * months / 12)

    return df


def _fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values before feeding to the model.

    Strategy by column:
    - boundary_layer_m: often missing from Open-Meteo — fill with 1000m
      (reasonable daytime average for Indian plains)
    - pressure_hpa: fill with standard sea-level pressure 1013.25 hPa
    - aod: forward-fill up to 3h gap (cloud gaps); beyond that use column median
    - Everything else: forward-fill then backward-fill for short gaps
    """
    df = df.copy()

    # Column-specific fills
    if "boundary_layer_m" in df.columns:
        df["boundary_layer_m"] = df["boundary_layer_m"].fillna(1000.0)

    if "pressure_hpa" in df.columns:
        df["pressure_hpa"] = df["pressure_hpa"].fillna(1013.25)

    # AOD — forward fill gaps up to 3h, then fill remaining with median
    if "aod" in df.columns:
        df["aod"] = df["aod"].ffill(limit=3)
        df["aod"] = df["aod"].fillna(df["aod"].median())

    # Generic fill for any remaining NaNs in feature columns
    present_features = [c for c in FEATURE_COLS if c in df.columns]
    df[present_features] = (
        df[present_features]
        .ffill(limit=2)
        .bfill(limit=2)
    )

    return df


def run_proxy_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main inference function — takes the merged ingestion DataFrame
    and returns it with two new columns: pm25_proxy and pm10_proxy.

    Parameters
    ----------
    df : pd.DataFrame
        Output from ingestion/pipeline.py — hourly rows with satellite
        and weather columns. Must have a UTC DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with pm25_proxy (µg/m³) and pm10_proxy (µg/m³)
        columns added. Rows where AOD was completely unavailable will
        have NaN in both columns.

    Raises
    ------
    ModelNotFoundError
        If the .joblib artifacts haven't been trained yet.
    InsufficientFeaturesError
        If none of the required feature columns are present.
    """
    # Verify artifacts exist before doing any work
    pm25_model, pm10_model, scaler = _load_artifacts()

    # Check we at least have AOD and basic weather
    missing = [c for c in ["aod", "temp_c", "humidity_pct"] if c not in df.columns]
    if missing:
        raise InsufficientFeaturesError(
            f"Proxy model can't run — missing columns: {missing}. "
            "Make sure the ingestion pipeline ran successfully."
        )

    # Add cyclic time features from the DatetimeIndex
    df = _add_time_features(df)

    # Fill missing values
    df = _fill_missing_features(df)

    # Identify rows where we have enough data to predict
    # (need at least AOD + temp + humidity to be non-null)
    core_cols   = ["aod", "temp_c", "humidity_pct"]
    predictable = df[core_cols].notna().all(axis=1)

    n_predictable = predictable.sum()
    n_total       = len(df)
    logger.info(
        "Running proxy inference on {}/{} rows ({:.0f}% coverage)",
        n_predictable, n_total, 100 * n_predictable / n_total
    )

    # Initialise output columns with NaN
    df["pm25_proxy"] = np.nan
    df["pm10_proxy"] = np.nan

    if n_predictable == 0:
        logger.warning(
            "No rows had enough data for proxy inference. "
            "Satellite coverage was likely 0% — check GEE pull."
        )
        return df

    # Build feature matrix for predictable rows only
    X = df.loc[predictable, FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    # Predict and clip to physically reasonable ranges
    pm25_pred = pm25_model.predict(X_scaled).clip(0, 500)
    pm10_pred = pm10_model.predict(X_scaled).clip(0, 700)

    df.loc[predictable, "pm25_proxy"] = pm25_pred
    df.loc[predictable, "pm10_proxy"] = pm10_pred

    logger.info(
        "Proxy estimates — PM2.5: mean={:.1f}, max={:.1f} µg/m³ | "
        "PM10: mean={:.1f}, max={:.1f} µg/m³",
        np.nanmean(pm25_pred), np.nanmax(pm25_pred),
        np.nanmean(pm10_pred), np.nanmax(pm10_pred),
    )

    return df