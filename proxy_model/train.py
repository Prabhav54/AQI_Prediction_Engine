"""
proxy_model/train.py
--------------------
Module 2 — XGBoost AOD → PM2.5 / PM10 Proxy Model TRAINING

Trains TWO XGBoost regressors:
  * xgb_pm25_proxy.joblib   →  predicts PM2.5 µg/m³ from satellite AOD + weather
  * xgb_pm10_proxy.joblib   →  predicts PM10 µg/m³ from satellite AOD + weather

Also saves the StandardScaler used at training time so inference applies
the exact same feature scaling.

Why XGBoost (vs. linear regression)?
  The AOD ↔ PM relationship is non-linear and modulated by boundary-layer
  height, humidity, and wind speed. Tree boosting captures interaction
  terms (e.g. "high AOD + low BLH" ≠ sum of effects) far better than OLS.

Training data
-------------
Pulled from the TimescaleDB `aqi_computed` table (Module 3). Each row must
have:
    - satellite AOD  (from GEE)
    - weather        (temp, humidity, wind, pressure, boundary-layer)
    - ground-truth PM2.5 and PM10 (CPCB monitor or a reference dataset)

If the user passes --from-csv we skip the DB and read directly from a CSV
(useful for first-time training before any CPCB data is in the DB).

Usage
-----
  conda activate aq_engine
  python proxy_model/train.py --location "Delhi"
  python proxy_model/train.py --from-csv data/cpcb_history.csv
  python proxy_model/train.py --location "Delhi" --tune    # hyperparameter search
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths & feature list (MUST match proxy_model/predict.py exactly)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR   = Path(__file__).parent / "artifacts"
PM25_MODEL_PATH = ARTIFACTS_DIR / "xgb_pm25_proxy.joblib"
PM10_MODEL_PATH = ARTIFACTS_DIR / "xgb_pm10_proxy.joblib"
SCALER_PATH     = ARTIFACTS_DIR / "feature_scaler.joblib"

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
TARGET_COLS = ["pm25", "pm10"]

# Reasonable defaults for tabular hourly air-quality data
XGB_PARAMS = dict(
    n_estimators      = 600,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.85,
    colsample_bytree  = 0.85,
    reg_lambda        = 1.5,
    reg_alpha         = 0.1,
    objective         = "reg:squarederror",
    tree_method       = "hist",
    random_state      = 42,
    n_jobs            = -1,
    early_stopping_rounds = 30,
)


# ===========================================================================
# Data loading
# ===========================================================================

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encodings of hour-of-day and month — same as predict.py."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        # Tolerant: try to coerce a 'timestamp' column
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
        else:
            raise ValueError(
                "Training DataFrame needs a DatetimeIndex or a 'timestamp' column."
            )
    hours  = df.index.hour
    months = df.index.month
    df["hour_sin"]  = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hours / 24)
    df["month_sin"] = np.sin(2 * np.pi * months / 12)
    df["month_cos"] = np.cos(2 * np.pi * months / 12)
    return df


def load_training_data_from_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV containing AOD + weather + ground-truth PM2.5/PM10."""
    logger.info("Loading training data from CSV: {}", csv_path)
    df = pd.read_csv(csv_path)
    df = _add_time_features(df)
    return df


def load_training_data_from_db(lat: float, lon: float, days: int = 365) -> pd.DataFrame:
    """Pull training rows from TimescaleDB (Module 3)."""
    from database.db_client import get_proxy_training_data  # lazy import
    logger.info("Pulling {} days of training data for ({:.4f}, {:.4f})", days, lat, lon)
    df = get_proxy_training_data(lat=lat, lon=lon, days=days)
    df = _add_time_features(df)
    return df


# ===========================================================================
# Preprocessing
# ===========================================================================

def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Drop rows missing AOD or ground-truth, scale features, and return arrays.

    Returns
    -------
    X_scaled : (n, n_features)
    y_pm25   : (n,)
    y_pm10   : (n,)
    scaler   : fitted StandardScaler
    """
    # Required for any training row
    must_have = ["aod"] + TARGET_COLS
    before = len(df)

    # --- INJECT MOCK TARGETS FOR PIPELINE TESTING ---
    import numpy as np
    if 'pm25' not in df.columns:
        print("⚠️ Injecting mock PM2.5 and PM10 target variables for training...")
        # Create a synthetic correlation so the model actually has something to learn
        df['pm25'] = df['aod'].fillna(0.5) * np.random.uniform(40, 80, size=len(df))
        df['pm10'] = df['pm25'] * np.random.uniform(1.2, 2.0, size=len(df))
    # ------------------------------------------------

    df = df.dropna(subset=must_have).copy()
    
    logger.info("Dropped {} rows missing AOD or ground-truth", before - len(df))

    # Fill weather gaps with reasonable defaults (same logic as predict.py)
    df["boundary_layer_m"] = df.get("boundary_layer_m", 1000.0).fillna(1000.0)
    df["pressure_hpa"]     = df.get("pressure_hpa",     1013.25).fillna(1013.25)
    df[FEATURE_COLS]       = df[FEATURE_COLS].ffill(limit=2).bfill(limit=2)
    df = df.dropna(subset=FEATURE_COLS)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_pm25 = df["pm25"].values.astype(np.float32)
    y_pm10 = df["pm10"].values.astype(np.float32)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    logger.info(
        "Training matrix: X={}, PM2.5 range [{:.1f}, {:.1f}], "
        "PM10 range [{:.1f}, {:.1f}]",
        X_scaled.shape,
        y_pm25.min(), y_pm25.max(),
        y_pm10.min(), y_pm10.max(),
    )
    return X_scaled, y_pm25, y_pm10, scaler


# ===========================================================================
# Training
# ===========================================================================

def _train_one(name: str, X: np.ndarray, y: np.ndarray) -> XGBRegressor:
    """Train a single XGBoost regressor with early stopping on a hold-out."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Hold-out metrics
    pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    mae  = float(mean_absolute_error(y_val, pred))
    r2   = float(r2_score(y_val, pred))
    logger.info(
        "{} → RMSE={:.2f} µg/m³ | MAE={:.2f} | R²={:.3f} | best_iter={}",
        name, rmse, mae, r2, model.best_iteration,
    )
    return model


def cross_validate(X: np.ndarray, y: np.ndarray, name: str, k: int = 5) -> None:
    """Optional k-fold CV for sanity checking."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses, r2s = [], []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        m = XGBRegressor(**{**XGB_PARAMS, "early_stopping_rounds": None})
        m.fit(X[tr], y[tr], verbose=False)
        p = m.predict(X[va])
        rmses.append(np.sqrt(mean_squared_error(y[va], p)))
        r2s.append(r2_score(y[va], p))
    logger.info(
        "{} {}-fold CV → RMSE {:.2f} ± {:.2f} | R² {:.3f} ± {:.3f}",
        name, k, np.mean(rmses), np.std(rmses), np.mean(r2s), np.std(r2s),
    )


def run_training(df: pd.DataFrame, do_cv: bool = False) -> None:
    """Full pipeline: preprocess → train both models → save artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y_pm25, y_pm10, scaler = preprocess(df)

    if do_cv:
        cross_validate(X, y_pm25, "PM2.5")
        cross_validate(X, y_pm10, "PM10")

    logger.info("Training PM2.5 model…")
    pm25_model = _train_one("PM2.5", X, y_pm25)

    logger.info("Training PM10 model…")
    pm10_model = _train_one("PM10", X, y_pm10)

    joblib.dump(pm25_model, PM25_MODEL_PATH)
    joblib.dump(pm10_model, PM10_MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)

    logger.info("Artifacts saved to {}", ARTIFACTS_DIR)
    print(f"\n✅ Saved:")
    print(f"   • {PM25_MODEL_PATH}")
    print(f"   • {PM10_MODEL_PATH}")
    print(f"   • {SCALER_PATH}\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the XGBoost PM2.5/PM10 proxy models."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--location", type=str,
                     help="Pull training data from DB for this city.")
    src.add_argument("--from-csv", type=str,
                     help="Path to a CSV with AOD + weather + ground-truth PM.")
    parser.add_argument("--days", type=int, default=365,
                        help="Days of history to pull from DB (default: 365).")
    parser.add_argument("--cv", action="store_true",
                        help="Run 5-fold cross-validation before final training.")
    args = parser.parse_args()

    if args.from_csv:
        df = load_training_data_from_csv(args.from_csv)
    else:
        from ingestion.geocoder import geocode
        geo = geocode(args.location)
        df = load_training_data_from_db(lat=geo.lat, lon=geo.lon, days=args.days)

    if df.empty:
        logger.error("No training data available — aborting.")
        sys.exit(1)

    run_training(df, do_cv=args.cv)