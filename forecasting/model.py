"""
forecasting/model.py
--------------------
Module 4 — LSTM + GRU multi-horizon forecasters.

Both models share one architecture (MQ-RNN style, Wen et al. 2017):

    past 168h ──► 4h patches (42 steps) ──► LSTM/GRU encoder ──► context h
                                         │  (repeated for each of 24 steps)
    future weather t+1..t+24 ──► per-step MLP ──┤
    horizon embedding ─────────────────────────┤
                                         ▼
                                   MLP decoder ──► 6 pollutants × 24h

Patching (as in PatchTST, Nie et al. 2023) concatenates every 4 consecutive
hours into one encoder step: the RNN still sees the full 168h but unrolls
42 steps instead of 168 — ~50x faster on CPU with no loss of history.

Why this instead of autoregressive rollout:
  * Direct multi-horizon output — no compounding error from feeding
    predictions back in.
  * Known future covariates — forecast weather (wind, boundary layer,
    rain) is the main driver of tomorrow's pollution, and Open-Meteo gives
    it to us for free. The old rollout held weather constant for 24h.

LSTMs and GRUs make different mistakes on the same input; blending them
with LightGBM (forecasting/ensemble.py) reduces variance.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exceptions import CheckpointNotFoundError
from forecasting.dataset import FUTURE_FEATURES, HORIZON, PAST_FEATURES, FeatureSpec
from config.settings import POLLUTANTS
from logger import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
LSTM_CKPT_PATH = CHECKPOINT_DIR / "lstm.pt"
GRU_CKPT_PATH = CHECKPOINT_DIR / "gru.pt"
LGBM_PATH = CHECKPOINT_DIR / "lgbm.joblib"
SPEC_PATH = CHECKPOINT_DIR / "feature_spec.json"
WEIGHTS_PATH = CHECKPOINT_DIR / "ensemble_weights.json"
MODEL_CARD_PATH = CHECKPOINT_DIR / "model_card.json"


# ===========================================================================
# Architecture
# ===========================================================================

class _RecurrentForecaster(nn.Module):
    """Shared scaffolding for the LSTM and GRU forecasters."""

    def __init__(
        self,
        rnn_cls,
        n_features: int = len(PAST_FEATURES),
        n_future: int = len(FUTURE_FEATURES),
        horizon: int = HORIZON,
        n_targets: int = len(POLLUTANTS),
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        patch_len: int = 4,
    ):
        super().__init__()
        self.config = dict(n_features=n_features, n_future=n_future, horizon=horizon,
                           n_targets=n_targets, hidden_size=hidden_size, n_layers=n_layers,
                           dropout=dropout, patch_len=patch_len)
        self.horizon = horizon
        self.patch_len = patch_len
        self.rnn = rnn_cls(
            input_size=n_features * patch_len,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.future_enc = nn.Sequential(nn.Linear(n_future, 32), nn.GELU())
        self.horizon_emb = nn.Embedding(horizon, 16)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + 32 + 16, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_targets),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        B, T, F = past.shape
        past = past.reshape(B, T // self.patch_len, F * self.patch_len)   # (B, 42, 4F)
        out, _ = self.rnn(past)                                  # (B, 42, H)
        ctx = out[:, -1, :].unsqueeze(1).expand(-1, self.horizon, -1)   # (B, 24, H)
        f = self.future_enc(future)                              # (B, 24, 32)
        steps = torch.arange(self.horizon, device=past.device)
        e = self.horizon_emb(steps).unsqueeze(0).expand(past.size(0), -1, -1)  # (B, 24, 16)
        return self.decoder(torch.cat([ctx, f, e], dim=-1))      # (B, 24, 6)


class AQIForecastLSTM(_RecurrentForecaster):
    """Stacked LSTM encoder."""

    def __init__(self, **kwargs):
        super().__init__(nn.LSTM, **kwargs)


class AQIForecastGRU(_RecurrentForecaster):
    """Stacked GRU encoder — lighter than LSTM, often better on noisy series."""

    def __init__(self, **kwargs):
        super().__init__(nn.GRU, **kwargs)


_MODELS = {"lstm": AQIForecastLSTM, "gru": AQIForecastGRU}


def artifact_paths(ckpt_dir: Path = CHECKPOINT_DIR) -> dict[str, Path]:
    """Every artifact of one model generation (champion or candidate) lives in one directory."""
    d = Path(ckpt_dir)
    return {
        "lstm": d / LSTM_CKPT_PATH.name, "gru": d / GRU_CKPT_PATH.name, "lgbm": d / LGBM_PATH.name,
        "spec": d / SPEC_PATH.name, "weights": d / WEIGHTS_PATH.name, "card": d / MODEL_CARD_PATH.name,
    }


# ===========================================================================
# Checkpoint save / load
# ===========================================================================

def save_checkpoint(model: _RecurrentForecaster, model_type: str, epoch: int, val_loss: float,
                    ckpt_dir: Path = CHECKPOINT_DIR) -> None:
    """Persist weights + architecture config. Scalers live in feature_spec.json."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    target = artifact_paths(ckpt_dir)[model_type]
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "config": model.config,
        "model_class": model.__class__.__name__,
    }, target)
    logger.info("Checkpoint saved → {} (epoch={}, val_loss={:.4f})", target, epoch, val_loss)


def load_checkpoint(device: torch.device, model_type: str = "lstm",
                    ckpt_dir: Path = CHECKPOINT_DIR) -> _RecurrentForecaster:
    ctor = _MODELS[model_type.lower()]
    path = artifact_paths(ckpt_dir)[model_type.lower()]
    if not path.exists():
        raise CheckpointNotFoundError(
            f"{model_type.upper()} checkpoint not found at {path}. "
            f"Run: python forecasting/train.py --model {model_type}"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ctor(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info("{} checkpoint loaded (epoch={}, val_loss={:.4f})",
                model_type.upper(), ckpt["epoch"], ckpt["val_loss"])
    return model


def load_spec(ckpt_dir: Path = CHECKPOINT_DIR) -> FeatureSpec:
    path = artifact_paths(ckpt_dir)["spec"]
    if not path.exists():
        raise CheckpointNotFoundError(f"Feature spec missing at {path}. Train first.")
    return FeatureSpec.from_json(path)
