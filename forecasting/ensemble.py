"""
forecasting/ensemble.py
-----------------------
Module 4 — LSTM + GRU + LightGBM forecast ensemble (batch inference).

  1. Loads the LSTM and GRU checkpoints and the LightGBM forecaster
     (one model per pollutant, horizon as a feature).
  2. Predicts 24h × 6 pollutant log-changes for every node in one batch.
  3. Blends members with per-pollutant weights found on the validation
     split (checkpoints/ensemble_weights.json — rewritten by every retrain).
  4. Adds 80% intervals from validation residual quantiles (split-conformal).
  5. Converts the forecast concentrations to CPCB AQI along the path
     [last 23 observed hours + forecast], exactly as refresh_aqi() would.

Members that are missing are skipped and the weights renormalised.
"""

import json
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from config.settings import POLLUTANTS
from exceptions import CheckpointNotFoundError
from forecasting.dataset import (
    HORIZON,
    FeatureSpec,
    build_store,
    history_conc,
    last_log,
    lgbm_features,
    nn_batch,
    normalise_store,
)
from forecasting.model import CHECKPOINT_DIR, artifact_paths, load_checkpoint, load_spec
from logger import get_logger
from utils import aqi_along_paths, aqi_category

logger = get_logger(__name__)

MEMBERS = ("lstm", "gru", "lgbm")
DEFAULT_WEIGHTS = {"lstm": 0.4, "gru": 0.3, "lgbm": 0.3}
_LABELS = np.array(["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])


# ===========================================================================
# Loading
# ===========================================================================

def load_ensemble_weights(ckpt_dir=CHECKPOINT_DIR) -> dict[str, dict[str, float]]:
    """{pollutant: {member: weight}} — falls back to DEFAULT_WEIGHTS for every pollutant."""
    path = artifact_paths(ckpt_dir)["weights"]
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as e:
            logger.warning("Could not parse {} ({}); using defaults.", path, e)
    return {p: dict(DEFAULT_WEIGHTS) for p in POLLUTANTS}


def load_members(device: torch.device, ckpt_dir=CHECKPOINT_DIR) -> dict:
    members = {}
    for name in ("lstm", "gru"):
        try:
            members[name] = load_checkpoint(device, name, ckpt_dir)
        except CheckpointNotFoundError as e:
            logger.warning("{}", e)
    lgbm_path = artifact_paths(ckpt_dir)["lgbm"]
    if lgbm_path.exists():
        members["lgbm"] = joblib.load(lgbm_path)
    if not members:
        raise CheckpointNotFoundError("No trained ensemble members found. Run forecasting/train.py.")
    return members


def load_card(ckpt_dir=CHECKPOINT_DIR) -> dict:
    path = artifact_paths(ckpt_dir)["card"]
    return json.loads(path.read_text()) if path.exists() else {}


# ===========================================================================
# Prediction primitives (shared with training / evaluation)
# ===========================================================================

def predict_nn(model, store, index, spec: FeatureSpec, device, batch: int = 512) -> np.ndarray:
    """(N, 24, 6) unscaled log-deltas."""
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(index), batch):
            past, fut = nn_batch(store, index[i:i + batch])
            pred = model(torch.from_numpy(past).to(device), torch.from_numpy(fut).to(device))
            outs.append(pred.cpu().numpy())
    return np.concatenate(outs) * spec.target_std


def predict_lgbm(models: dict, store, index, spec: FeatureSpec) -> np.ndarray:
    """(N, 24, 6) unscaled log-deltas from the per-pollutant LightGBM models."""
    rep = np.repeat(index, HORIZON, axis=0)
    hs = np.tile(np.arange(1, HORIZON + 1), len(index))
    X = lgbm_features(store, rep, hs)
    out = np.zeros((len(index), HORIZON, len(POLLUTANTS)), dtype=np.float32)
    for j, p in enumerate(POLLUTANTS):
        out[:, :, j] = models[p].predict(X).reshape(len(index), HORIZON) * spec.target_std[:, j]
    return out


def predict_members(members: dict, store, index, spec, device) -> dict[str, np.ndarray]:
    preds = {}
    for name, m in members.items():
        preds[name] = predict_lgbm(m, store, index, spec) if name == "lgbm" \
            else predict_nn(m, store, index, spec, device)
    return preds


def blend(preds: dict[str, np.ndarray], weights: dict[str, dict[str, float]]) -> np.ndarray:
    """Per-pollutant weighted average over whichever members are present."""
    names = list(preds)
    out = np.zeros_like(preds[names[0]])
    for j, p in enumerate(POLLUTANTS):
        w = np.array([weights.get(p, DEFAULT_WEIGHTS).get(n, 0.0) for n in names])
        w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1 / len(names))
        out[:, :, j] = sum(wi * preds[n][:, :, j] for wi, n in zip(w, names))
    return out


def deltas_to_conc(last: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """log1p(y_t) + Δ → concentrations (N, 24, 6)."""
    return np.expm1(np.clip(last[:, None, :] + deltas, 0, None))


# ===========================================================================
# Public API — forecast every node
# ===========================================================================

def forecast_all_nodes(nodes: pd.DataFrame, frame: pd.DataFrame,
                       issue_time: Optional[datetime] = None, ckpt_dir=CHECKPOINT_DIR) -> pd.DataFrame:
    """
    24h forecast for every node with enough history.

    `frame` is the ingestion output: hourly rows for [now - 8 days, now + 48h],
    where the future rows carry Open-Meteo *forecast* weather.
    Returns rows matching the aqi_forecasts table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_spec(ckpt_dir)
    members = load_members(device, ckpt_dir)
    weights = load_ensemble_weights(ckpt_dir)
    card = load_card(ckpt_dir)

    issue = pd.Timestamp(issue_time or datetime.now(timezone.utc)).floor("h")
    store = build_store(frame, nodes, require_targets=False)

    index = []
    for n, s in enumerate(store):
        if issue in s.times:
            t = s.times.get_loc(issue)
            if s.valid[t]:
                index.append((n, t))
    if not index:
        raise CheckpointNotFoundError(f"No node has a complete window at issue time {issue}.")
    index = np.asarray(index)
    logger.info("Forecasting {} / {} nodes at issue time {}", len(index), len(store), issue)

    norm = normalise_store(store, spec)
    preds = predict_members(members, norm, index, spec, device)
    ens = blend(preds, weights)

    last = last_log(store, index)
    hist = history_conc(store, index)
    conc = deltas_to_conc(last, ens)
    aqi, prom = aqi_along_paths(hist, conc)

    # 80% interval from validation residual quantiles of the blended forecast
    q = card.get("residual_quantiles")
    if q:
        lo = deltas_to_conc(last, ens + np.asarray(q["q10"])[None])
        hi = deltas_to_conc(last, ens + np.asarray(q["q90"])[None])
        aqi_lo, _ = aqi_along_paths(hist, lo)
        aqi_hi, _ = aqi_along_paths(hist, hi)
    else:
        aqi_lo = aqi_hi = np.full_like(aqi, np.nan)

    member_aqi = {k: aqi_along_paths(hist, deltas_to_conc(last, v))[0] for k, v in preds.items()}

    version = card.get("model_version", "untrained")
    rows = []
    for i, (n, t) in enumerate(index):
        h_ = store[n].location_hash
        for h in range(HORIZON):
            rows.append({
                "forecast_generated_at": issue.to_pydatetime(),
                "forecast_target_time": (issue + pd.Timedelta(hours=h + 1)).to_pydatetime(),
                "location_hash": h_,
                "hours_ahead": h + 1,
                **{p: float(conc[i, h, j]) for j, p in enumerate(POLLUTANTS)},
                "aqi": int(round(aqi[i, h])),
                "aqi_lower": None if np.isnan(aqi_lo[i, h]) else int(round(aqi_lo[i, h])),
                "aqi_upper": None if np.isnan(aqi_hi[i, h]) else int(round(aqi_hi[i, h])),
                "aqi_category": aqi_category(int(round(aqi[i, h]))),
                "prominent_pollutant": _LABELS[prom[i, h]],
                **{f"aqi_{k}": int(round(member_aqi[k][i, h])) if k in member_aqi else None
                   for k in MEMBERS},
                "model_version": version,
            })
    out = pd.DataFrame(rows)
    logger.info("Ensemble forecast | members={} | AQI range [{} – {}]",
                list(preds), out["aqi"].min(), out["aqi"].max())
    return out
