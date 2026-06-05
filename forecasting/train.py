"""
forecasting/train.py
--------------------
Module 4 — Trainer for the LSTM/Transformer + GRU + LightGBM ensemble.

Trains (any combination of):
  • LSTM/Transformer -> forecasting/checkpoints/lstm_aqi.pt
  • GRU              -> forecasting/checkpoints/gru_aqi.pt
  • LightGBM         -> forecasting/checkpoints/xgb_forecaster.joblib (Kept naming for API compatibility)

Then runs an OOF blend search to write best weights into
forecasting/checkpoints/ensemble_weights.json.
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Subset

# --- PHASE 2 UPGRADE: LightGBM ---
import lightgbm as lgb
from lightgbm import LGBMRegressor

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

# --- PHASE 3 SAFE UPGRADE ---
# This safely checks if you have added the Transformer to model.py yet.
try:
    from forecasting.model import AQIForecastTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

from ingestion.geocoder import geocode
from logger import get_logger

logger = get_logger(__name__)

# Shared hyperparameters
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 15
DEFAULT_EPOCHS = 80

# Keep legacy names so the FastAPI backend doesn't crash
XGB_FORECAST_PATH = CHECKPOINT_DIR / "xgb_forecaster.joblib"
ENS_WEIGHTS_PATH  = CHECKPOINT_DIR / "ensemble_weights.json"
XGB_TAIL_HOURS    = 24


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
            flag = "  <- best"
        else:
            stale += 1

        if ep % 5 == 0 or ep == 1:
            print(f"  {name} {ep:3d}/{epochs} | train={tr:.4f} | val={vl:.4f}{flag}")

        if stale >= PATIENCE:
            print(f"\n  Early stopping {name} at epoch {ep}.")
            break

    print(f"\n  {name} best val loss: {best:.4f}\n")
    return best


# ===========================================================================
# LightGBM forecaster (tabular flatten of recent window)
# ===========================================================================

def _build_xgb_dataset(X_seq: np.ndarray, y_seq: np.ndarray, tail: int = XGB_TAIL_HOURS):
    Xt = X_seq[:, -tail:, :].reshape(X_seq.shape[0], -1)
    return Xt, y_seq


def _train_lgb(X_seq, y_seq, n_train):
    X_tab, y_tab = _build_xgb_dataset(X_seq, y_seq)
    X_tr, y_tr = X_tab[:n_train], y_tab[:n_train]
    X_va, y_va = X_tab[n_train:], y_tab[n_train:]

    # LightGBM handles tabular time-series features better and faster
    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_tr, y_tr, 
        eval_set=[(X_va, y_va)], 
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )

    val_rmse = float(np.sqrt(mean_squared_error(y_va, model.predict(X_va))))
    logger.info("LightGBM forecaster val RMSE (normalised): {:.4f}", val_rmse)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, XGB_FORECAST_PATH)
    print(f"  LightGBM forecaster saved -> {XGB_FORECAST_PATH}\n")
    return model


# ===========================================================================
# Ensemble weight search
# ===========================================================================

def _search_weights(lstm_val, gru_val, xgb_val, y_val) -> dict:
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
    print(f"  Ensemble weights saved -> {ENS_WEIGHTS_PATH}: {best_w}\n")
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
        if TRANSFORMER_AVAILABLE:
            logger.info("Phase 3 Detected: Using state-of-the-art PyTorch Transformer")
            # We keep the name "LSTM" so the checkpoint saves correctly for the API
            _train_nn("LSTM", AQIForecastTransformer, train_loader, val_loader,
                      X.shape[2], scaler_params, device, epochs)
        else:
            logger.info("Using standard LSTM. (Add AQIForecastTransformer to model.py to upgrade!)")
            _train_nn("LSTM", AQIForecastLSTM, train_loader, val_loader,
                      X.shape[2], scaler_params, device, epochs)
            
    if want_gru:
        _train_nn("GRU",  AQIForecastGRU,  train_loader, val_loader,
                  X.shape[2], scaler_params, device, epochs)
        
    if want_xgb:
        _train_lgb(X, y, n_train)

    if which == "all":
        from forecasting.model import load_checkpoint
        lstm_model, _ = load_checkpoint(device, "lstm")
        gru_model,  _ = load_checkpoint(device, "gru")
        lgb_model     = joblib.load(XGB_FORECAST_PATH)

        y_val = y[n_train:]
        lstm_val = _nn_val_predictions(lstm_model, val_loader, device)
        gru_val  = _nn_val_predictions(gru_model,  val_loader, device)
        X_tab, _ = _build_xgb_dataset(X, y)
        lgb_val  = lgb_model.predict(X_tab[n_train:])

        m = min(len(y_val), len(lstm_val), len(gru_val), len(lgb_val))
        _search_weights(lstm_val[:m], gru_val[:m], lgb_val[:m], y_val[:m])


class AQIForecastTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 1. Expand features
        self.input_linear = nn.Linear(n_features, d_model)
        
        # 2. Time Position Awareness (Crucial for predicting rush hour)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 168, d_model)) 
        
        # 3. The Attention Brain
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final output layer
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_linear(x) 
        x = x + self.pos_encoder[:, :x.size(1), :] 
        x = self.transformer(x)
        out = self.output_linear(x[:, -1, :]) 
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--model", choices=["all", "lstm", "gru", "xgb"], default="all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    geo = geocode(args.location)
    run_training(lat=geo.lat, lon=geo.lon, location_name=geo.display_name,
                 which=args.model, epochs=args.epochs)