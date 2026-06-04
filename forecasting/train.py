"""
forecasting/train.py
--------------------
Module 4 — Trainer for the LSTM + GRU + XGBoost forecasting ensemble.

Trains (any combination of):
  • LSTM   →  forecasting/checkpoints/lstm_aqi.pt
  • GRU    →  forecasting/checkpoints/gru_aqi.pt
  • XGB    →  forecasting/checkpoints/xgb_forecaster.joblib

Then runs an OOF blend search to write best weights into
forecasting/checkpoints/ensemble_weights.json.

Usage
-----
  conda activate aq_engine

  # Train everything (recommended)
  python forecasting/train.py --location "Delhi"

  # Train just one component
  python forecasting/train.py --location "Mumbai" --model lstm
  python forecasting/train.py --location "Mumbai" --model gru
  python forecasting/train.py --location "Mumbai" --model xgb

Notes
-----
* Time-ordered 80/20 train/val split (no shuffling) so validation can't
  see the future.
* Early stopping at 15 stale epochs for the NN models; XGBoost uses its
  own internal early stopping on the same val split.
"""

import argparse
import json
import sys
from pathlib import Path
from sklearn.model_selection import GridSearchCV

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Subset
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_client import get_lstm_input_sequence
from exceptions import SequenceTooShortError
from forecasting.dataset import AQISequenceDataset, prepare_sequence
from forecasting.model import (
    AQIForecastGRU,
    AQIForecastLSTM,
    CHECKPOINT_DIR,
    save_checkpoint,
)
from ingestion.geocoder import geocode
from logger import get_logger

logger = get_logger(__name__)

# Shared hyperparameters
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 15
DEFAULT_EPOCHS = 80

XGB_FORECAST_PATH = CHECKPOINT_DIR / "xgb_forecaster.joblib"
ENS_WEIGHTS_PATH  = CHECKPOINT_DIR / "ensemble_weights.json"
XGB_TAIL_HOURS    = 24   # how many recent hours to flatten as XGB input


# ===========================================================================
# Neural training helpers
# ===========================================================================

def _epoch(model, loader, optim, crit, device, train: bool) -> float:
    model.train() if train else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            pred = model(X)
            loss = crit(pred, y)
            if train:
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            total += loss.item()
    return total / max(len(loader), 1)


def _train_nn(name: str, model_cls, train_loader, val_loader, n_features,
              scaler_params, device, epochs: int):
    model = model_cls(n_features=n_features).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=5, factor=0.5,
    )
    crit  = nn.MSELoss()

    best, stale = float("inf"), 0
    print(f"\n  Training {name} | epochs={epochs} | device={device}\n")
    for ep in range(1, epochs + 1):
        tr = _epoch(model, train_loader, optim, crit, device, train=True)
        vl = _epoch(model, val_loader,   optim, crit, device, train=False)
        sched.step(vl)

        flag = ""
        if vl < best:
            best, stale = vl, 0
            save_checkpoint(model, scaler_params, ep, vl)
            flag = "  ← best"
        else:
            stale += 1

        if ep % 5 == 0 or ep == 1:
            print(f"  {name} {ep:3d}/{epochs} | "
                  f"train={tr:.4f} | val={vl:.4f}{flag}")

        if stale >= PATIENCE:
            print(f"\n  Early stopping {name} at epoch {ep}.")
            break

    print(f"\n  {name} best val loss: {best:.4f}\n")
    return best


# ===========================================================================
# XGBoost forecaster (tabular flatten of recent window)
# ===========================================================================

def _build_xgb_dataset(X_seq: np.ndarray, y_seq: np.ndarray, tail: int = XGB_TAIL_HOURS):
    """
    X_seq : (N, T, F)  sliding windows from prepare_sequence()
    y_seq : (N,)       AQI target one hour ahead (normalised)

    Returns
    -------
    X_tab : (N, tail * F)  last `tail` hours flattened
    y_tab : (N,)
    """
    Xt = X_seq[:, -tail:, :].reshape(X_seq.shape[0], -1)
    return Xt, y_seq


def _train_xgb(X_seq, y_seq, n_train) -> XGBRegressor:
    X_tab, y_tab = _build_xgb_dataset(X_seq, y_seq)
    X_tr, y_tr = X_tab[:n_train], y_tab[:n_train]
    X_va, y_va = X_tab[n_train:], y_tab[n_train:]

    logger.info("Starting XGBoost Grid Search to find optimal parameters...")

    # The Grid: The model will test every single combination of these values
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500, 800]
    }

    # Base model
    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    # Automated Scikit-learn search with 3-fold Cross Validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1
    )

    # Run the search on the training data
    grid_search.fit(X_tr, y_tr)

    # Extract the absolute best model it found
    best_model = grid_search.best_estimator_
    logger.info(f"Optimal XGBoost parameters found: {grid_search.best_params_}")

    # Final validation check
    val_rmse = float(np.sqrt(mean_squared_error(y_va, best_model.predict(X_va))))
    logger.info("Tuned XGB forecaster val RMSE (normalised): {:.4f}", val_rmse)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, XGB_FORECAST_PATH)
    print(f"  Tuned XGBoost forecaster saved → {XGB_FORECAST_PATH}\n")
    
    return best_model


# ===========================================================================
# Ensemble weight search
# ===========================================================================

def _search_weights(lstm_val, gru_val, xgb_val, y_val) -> dict:
    """Grid search 11x11x11 simplex of weights, minimise val RMSE."""
    best, best_w = float("inf"), None
    grid = np.linspace(0, 1, 11)
    for a in grid:
        for b in grid:
            c = 1.0 - a - b
            if c < 0 or c > 1:
                continue
            blend = a * lstm_val + b * gru_val + c * xgb_val
            rmse  = float(np.sqrt(mean_squared_error(y_val, blend)))
            if rmse < best:
                best, best_w = rmse, {"lstm": float(a), "gru": float(b), "xgb": float(c)}
    logger.info("Best ensemble weights: {} | val RMSE={:.4f}", best_w, best)
    ENS_WEIGHTS_PATH.write_text(json.dumps(best_w, indent=2))
    print(f"  Ensemble weights saved → {ENS_WEIGHTS_PATH}: {best_w}\n")
    return best_w


def _nn_val_predictions(model, val_loader, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in val_loader:
            preds.append(model(X.to(device)).cpu().numpy().squeeze(1))
    return np.concatenate(preds) if preds else np.array([])


# ===========================================================================
# Driver
# ===========================================================================

def run_training(lat: float, lon: float, location_name: str,
                 which: str = "all", epochs: int = DEFAULT_EPOCHS) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: {}", device)

    df = get_lstm_input_sequence(lat, lon, lookback_hours=720)
    if df.empty:
        logger.error("No data in DB for {}. Run the ingestion pipeline first.",
                     location_name)
        return

    try:
        X, y, scaler_params = prepare_sequence(df)
    except SequenceTooShortError as e:
        logger.error("{}", e)
        return

    dataset = AQISequenceDataset(X, y)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)

    train_set = Subset(dataset, list(range(n_train)))
    val_set   = Subset(dataset, list(range(n_train, n_total)))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Dataset: {} train / {} val | features={}",
                n_train, n_total - n_train, X.shape[2])

    want_lstm = which in ("all", "lstm")
    want_gru  = which in ("all", "gru")
    want_xgb  = which in ("all", "xgb")

    if want_lstm:
        _train_nn("LSTM", AQIForecastLSTM, train_loader, val_loader,
                  X.shape[2], scaler_params, device, epochs)
    if want_gru:
        _train_nn("GRU",  AQIForecastGRU,  train_loader, val_loader,
                  X.shape[2], scaler_params, device, epochs)
    if want_xgb:
        _train_xgb(X, y, n_train)

    # If we trained the whole stack, also pick optimal blending weights
    if which == "all":
        from forecasting.model import load_checkpoint
        lstm_model, _ = load_checkpoint(device, "lstm")
        gru_model,  _ = load_checkpoint(device, "gru")
        xgb_model     = joblib.load(XGB_FORECAST_PATH)

        y_val = y[n_train:]
        lstm_val = _nn_val_predictions(lstm_model, val_loader, device)
        gru_val  = _nn_val_predictions(gru_model,  val_loader, device)
        X_tab, _ = _build_xgb_dataset(X, y)
        xgb_val  = xgb_model.predict(X_tab[n_train:])

        # Defensive — keep the shortest common length in case of tail trimming
        m = min(len(y_val), len(lstm_val), len(gru_val), len(xgb_val))
        _search_weights(lstm_val[:m], gru_val[:m], xgb_val[:m], y_val[:m])


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the LSTM + GRU + XGBoost AQI forecasting ensemble."
    )
    parser.add_argument("--location", required=True,
                        help='City to train on, e.g. "Delhi" or "Mumbai, Maharashtra"')
    parser.add_argument("--model", choices=["all", "lstm", "gru", "xgb"],
                        default="all",
                        help="Train one component or all three (default).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Max epochs for NN models (default: {DEFAULT_EPOCHS}).")
    args = parser.parse_args()

    geo = geocode(args.location)
    logger.info("Training for: {}", geo)
    run_training(lat=geo.lat, lon=geo.lon, location_name=geo.display_name,
                 which=args.model, epochs=args.epochs)