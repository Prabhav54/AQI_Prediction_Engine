"""
forecasting/train.py
--------------------
Module 4 — LSTM Training Loop

Trains the AQI forecasting LSTM on historical data pulled from the
TimescaleDB aqi_computed table. Saves the best checkpoint (by
validation loss) to forecasting/checkpoints/lstm_aqi.pt.

What "best checkpoint" means here:
  We split the 168-hour sequences into 80% train / 20% validation
  (time-ordered — no shuffling, because leaking future data into
  training would give falsely optimistic results). We save the
  model from the epoch with the lowest validation RMSE.

Early stopping kicks in if validation loss doesn't improve for
15 consecutive epochs — saves time and prevents overfitting.

Usage:
  conda activate aq_engine
  python forecasting/train.py --location "Delhi"
  python forecasting/train.py --location "Mumbai" --epochs 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_client import get_lstm_input_sequence
from exceptions import SequenceTooShortError
from forecasting.dataset import AQISequenceDataset, prepare_sequence
from forecasting.model import AQIForecastLSTM, save_checkpoint
from ingestion.geocoder import geocode
from logger import get_logger

logger = get_logger(__name__)

# Training hyperparameters — reasonable defaults for city-level AQI
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4   # L2 regularisation via AdamW
PATIENCE        = 15     # early stopping patience (epochs)
DEFAULT_EPOCHS  = 80


def train_one_epoch(
    model:      AQIForecastLSTM,
    loader:     DataLoader,
    optimiser:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
) -> float:
    """Single training epoch. Returns average loss over all batches."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)  # (batch,) → (batch, 1)

        optimiser.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model:     AQIForecastLSTM,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    """Validation pass. Returns average loss, no gradient computation."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


def run_training(
    lat: float,
    lon: float,
    location_name: str,
    epochs: int = DEFAULT_EPOCHS,
) -> None:
    """
    Full training run for a single location.

    1. Pulls historical sequence from DB
    2. Builds sliding-window dataset
    3. Trains with early stopping
    4. Saves best checkpoint
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: {}", device)
    logger.info("Fetching training data for: {}", location_name)

    # Pull the sequence from DB — needs Module 1 + 3 to have run first
    df = get_lstm_input_sequence(lat, lon, lookback_hours=720)  # 30 days

    if df.empty:
        logger.error(
            "No data in DB for {}. Run the ingestion pipeline first: "
            "python ingestion/pipeline.py '{}'",
            location_name, location_name
        )
        return

    # Build dataset
    try:
        X, y, scaler_params = prepare_sequence(df)
    except SequenceTooShortError as e:
        logger.error("{}", e)
        return

    dataset = AQISequenceDataset(X, y)

    # Time-ordered 80/20 split — no shuffling
    n_total    = len(dataset)
    n_train    = int(0.8 * n_total)
    n_val      = n_total - n_train

    # Use Subset rather than random_split to preserve time ordering
    train_indices = list(range(n_train))
    val_indices   = list(range(n_train, n_total))

    from torch.utils.data import Subset
    train_set = Subset(dataset, train_indices)
    val_set   = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    logger.info(
        "Dataset: {} train / {} val samples | {} features",
        n_train, n_val, X.shape[2]
    )

    # Initialise model, optimiser, loss
    model     = AQIForecastLSTM().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    # ReduceLROnPlateau halves the learning rate if val loss stalls for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5, verbose=False
    )
    criterion = nn.MSELoss()   # MSE in normalised space ≈ RMSE in AQI units

    # Training loop with early stopping
    best_val_loss  = float("inf")
    patience_count = 0

    print(f"\n  Training LSTM for: {location_name}")
    print(f"  Epochs: {epochs} | Batch: {BATCH_SIZE} | Device: {device}\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Save checkpoint if this is the best epoch so far
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(model, scaler_params, epoch, val_loss)
            flag = "  ← best"
        else:
            patience_count += 1
            flag = ""

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f}{flag}"
            )

        # Early stopping
        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break

    print(f"\n  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint saved to: forecasting/checkpoints/lstm_aqi.pt\n")


# ================================================================
# Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the LSTM AQI forecasting model."
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help='City to train on, e.g. "Delhi" or "Mumbai, Maharashtra"'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Max training epochs (default: {DEFAULT_EPOCHS})."
    )
    args = parser.parse_args()

    # Geocode the location to get lat/lon for the DB query
    geo = geocode(args.location)
    logger.info("Training for: {}", geo)

    run_training(
        lat=geo.lat,
        lon=geo.lon,
        location_name=geo.display_name,
        epochs=args.epochs,
    )