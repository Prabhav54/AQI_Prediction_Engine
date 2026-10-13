"""
forecasting/train.py
--------------------
Module 4 — Trainer for the global LSTM + GRU + LightGBM ensemble.

One model generation is trained across every node at once:
  • LSTM      -> checkpoints/lstm.pt
  • GRU       -> checkpoints/gru.pt
  • LightGBM  -> checkpoints/lgbm.joblib   (one booster per pollutant, horizon as a feature)

Then, on held-out data:
  1. Per-pollutant blend weights are searched on the validation split
     (ensemble_weights.json).
  2. Validation residual quantiles give calibrated 80% intervals.
  3. Everything is scored on a later test split vs persistence / seasonal
     baselines and written to model_card.json.
  4. With --promote-if-better the new generation is trained into a
     candidate directory and only replaces the deployed champion if it
     beats it on the *same* test window (champion/challenger gate).

Usage
-----
  python forecasting/train.py --source parquet --data data/history.parquet
  python forecasting/train.py --source db --days 120 --promote-if-better
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import POLLUTANTS
from forecasting.dataset import (
    HORIZON,
    AQISequenceDataset,
    build_store,
    chronological_splits,
    fit_spec,
    lgbm_feature_names,
    lgbm_features,
    normalise_store,
    sample_index,
    true_deltas,
)
from forecasting.ensemble import blend, load_ensemble_weights, load_members, predict_members
from forecasting.evaluation import residual_quantiles, score
from forecasting.model import (
    CHECKPOINT_DIR,
    AQIForecastGRU,
    AQIForecastLSTM,
    artifact_paths,
    load_spec,
    save_checkpoint,
)
from logger import get_logger

logger = get_logger(__name__)

# Shared hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 3
DEFAULT_EPOCHS = 12
CANDIDATE_DIR = CHECKPOINT_DIR / "_candidate"
PROMOTION_TOLERANCE = 1.02   # candidate may be at most 2% worse (noise) and still promote


# ===========================================================================
# Data
# ===========================================================================

def load_frame(source: str, data: str, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nodes, long hourly frame) from the backfill parquet or the database."""
    if source == "parquet":
        frame = pd.read_parquet(data)
        nodes = frame[["location_hash", "lat", "lon"]].drop_duplicates("location_hash")
    else:
        from database.db_client import load_nodes, load_observations

        nodes = load_nodes()
        frame = load_observations(days)
        # Proxy-filled PM is a model output, not a target we want to learn
        gap = frame["pm_gap_filled"].fillna(False).astype(bool)
        frame.loc[gap, ["pm25", "pm10"]] = np.nan
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    logger.info("Loaded {} rows for {} nodes ({} → {})", len(frame), frame["location_hash"].nunique(),
                frame["timestamp"].min(), frame["timestamp"].max())
    return nodes, frame


def _subsample(index: np.ndarray, n: int | None, seed: int = 0) -> np.ndarray:
    if n is None or len(index) <= n:
        return index
    rng = np.random.default_rng(seed)
    return index[np.sort(rng.choice(len(index), n, replace=False))]


# ===========================================================================
# Neural members
# ===========================================================================

def _epoch(model, loader, optim, crit, device, train: bool) -> float:
    model.train() if train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for past, fut, y in loader:
            past, fut, y = past.to(device), fut.to(device), y.to(device)
            loss = crit(model(past, fut), y)
            if train:
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            total += loss.item() * len(y)
            n += len(y)
    return total / max(n, 1)


def _train_nn(name: str, model_cls, store, idx_tr, idx_va, spec, device, epochs: int,
              hidden: int, out_dir: Path) -> float:
    model = model_cls(hidden_size=hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", patience=1, factor=0.5)
    crit = nn.SmoothL1Loss()   # Huber — robust to dust-storm / stubble-burning spikes

    tr = DataLoader(AQISequenceDataset(store, idx_tr, spec), batch_size=BATCH_SIZE, shuffle=True)
    va = DataLoader(AQISequenceDataset(store, idx_va, spec), batch_size=BATCH_SIZE * 2)

    best, stale = float("inf"), 0
    logger.info("Training {} | {} train / {} val windows | epochs={} | device={}",
                name, len(idx_tr), len(idx_va), epochs, device)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        trl = _epoch(model, tr, optim, crit, device, train=True)
        vl = _epoch(model, va, optim, crit, device, train=False)
        sched.step(vl)
        flag = ""
        if vl < best:
            best, stale = vl, 0
            save_checkpoint(model, name.lower(), ep, vl, out_dir)
            flag = "  <- best"
        else:
            stale += 1
        logger.info("{} {:2d}/{} | train={:.4f} | val={:.4f} | {:.0f}s{}",
                    name, ep, epochs, trl, vl, time.time() - t0, flag)
        if stale >= PATIENCE:
            logger.info("Early stopping {} at epoch {}.", name, ep)
            break
    return best


# ===========================================================================
# LightGBM member
# ===========================================================================

def _lgbm_set(store, index, spec, horizons_per_sample: int, seed: int):
    """Each sample contributes a few random horizons — decorrelates rows, keeps the matrix small."""
    rng = np.random.default_rng(seed)
    rep = np.repeat(index, horizons_per_sample, axis=0)
    hs = rng.integers(1, HORIZON + 1, size=len(rep))
    X = lgbm_features(store, rep, hs)
    d = np.stack([store[n].ylog[t + h] - store[n].ylog[t] for (n, t), h in zip(rep, hs)])
    y = d / spec.target_std[hs - 1]
    return X, y


def _train_lgbm(store, idx_tr, idx_va, spec, out_dir: Path, n_samples: int) -> dict:
    t0 = time.time()
    X_tr, y_tr = _lgbm_set(store, _subsample(idx_tr, n_samples), spec, 6, seed=1)
    X_va, y_va = _lgbm_set(store, _subsample(idx_va, n_samples // 4), spec, 4, seed=2)
    logger.info("LightGBM matrices: train {} | val {} ({:.0f}s)", X_tr.shape, X_va.shape, time.time() - t0)

    names = lgbm_feature_names()
    models = {}
    for j, p in enumerate(POLLUTANTS):
        m = lgb.LGBMRegressor(
            objective="huber", alpha=1.0, n_estimators=1500, learning_rate=0.05,
            num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
        )
        m.fit(X_tr, y_tr[:, j], eval_set=[(X_va, y_va[:, j])], feature_name=names,
              callbacks=[lgb.early_stopping(50, verbose=False)])
        models[p] = m.booster_   # raw Booster: smaller pickle, predicts at best_iteration
        logger.info("LightGBM {:<4} best_iter={} val_huber={:.4f}",
                    p, m.best_iteration_, m.best_score_["valid_0"]["huber"])
    joblib.dump(models, artifact_paths(out_dir)["lgbm"], compress=3)
    return models


# ===========================================================================
# Ensemble weights
# ===========================================================================

def _search_weights(preds: dict[str, np.ndarray], true_d: np.ndarray, step: float = 0.05) -> dict:
    """Per-pollutant simplex grid search minimising validation MAE of log-changes."""
    names = list(preds)
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    combos = [c for c in np.array(np.meshgrid(*[grid] * len(names))).T.reshape(-1, len(names))
              if abs(c.sum() - 1) < 1e-6]
    weights = {}
    for j, p in enumerate(POLLUTANTS):
        stack = np.stack([preds[n][..., j] for n in names])          # (M, N, 24)
        maes = [np.abs(np.tensordot(c, stack, 1) - true_d[..., j]).mean() for c in combos]
        best = combos[int(np.argmin(maes))]
        weights[p] = {n: float(w) for n, w in zip(names, best)}
        logger.info("Blend weights {:<4}: {} (val MAE {:.4f})", p, weights[p], min(maes))
    return weights


# ===========================================================================
# Driver
# ===========================================================================

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def run_training(args) -> dict:
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes, frame = load_frame(args.source, args.data, args.days)
    store = build_store(frame, nodes)
    (_, tr_end), (va0, va1), (te0, _) = chronological_splits(store, args.val_days, args.test_days)

    idx_tr = _subsample(sample_index(store, args.stride, None, tr_end), args.max_train)
    idx_va = _subsample(sample_index(store, args.stride, va0, va1), args.max_eval)
    idx_te = _subsample(sample_index(store, args.stride, te0), args.max_eval, seed=3)
    logger.info("Splits | train <{} : {} | val {}→{} : {} | test ≥{} : {}",
                tr_end, len(idx_tr), va0, va1, len(idx_va), te0, len(idx_te))

    out_dir = CANDIDATE_DIR if args.promote_if_better else CHECKPOINT_DIR
    if out_dir.exists() and args.promote_if_better:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(out_dir)

    spec = fit_spec(store, idx_tr)
    spec.to_json(paths["spec"])
    norm = normalise_store(store, spec)

    t0 = time.time()
    if args.model in ("all", "lstm"):
        _train_nn("LSTM", AQIForecastLSTM, norm, idx_tr, idx_va, spec, device, args.epochs, args.hidden, out_dir)
    if args.model in ("all", "gru"):
        _train_nn("GRU", AQIForecastGRU, norm, idx_tr, idx_va, spec, device, args.epochs, args.hidden, out_dir)
    if args.model in ("all", "lgbm"):
        _train_lgbm(norm, idx_tr, idx_va, spec, out_dir, args.lgbm_samples)
    train_minutes = (time.time() - t0) / 60

    # ── Blend weights + intervals on validation ─────────────────────
    members = load_members(device, out_dir)
    val_preds = predict_members(members, norm, idx_va, spec, device)
    val_true = true_deltas(norm, idx_va)
    weights = _search_weights(val_preds, val_true)
    paths["weights"].write_text(json.dumps(weights, indent=2))
    val_ens = blend(val_preds, weights)
    quantiles = residual_quantiles(val_true, val_ens)

    # ── Held-out test ───────────────────────────────────────────────
    test_preds = predict_members(members, norm, idx_te, spec, device)
    test_preds["ensemble"] = blend(test_preds, weights)
    interval = {"lo": test_preds["ensemble"] + np.asarray(quantiles["q10"]),
                "hi": test_preds["ensemble"] + np.asarray(quantiles["q90"])}
    metrics = score(norm, idx_te, test_preds, interval)

    card = {
        "model_version": f"{datetime.now(timezone.utc):%Y.%m.%d}-{_git_sha()}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "source": args.source,
            "nodes": int(len(store)),
            "start": str(frame["timestamp"].min()),
            "end": str(frame["timestamp"].max()),
            "train_windows": int(len(idx_tr)), "val_windows": int(len(idx_va)),
            "test_windows": int(len(idx_te)),
            "test_period": [str(te0), str(frame["timestamp"].max())],
        },
        "architecture": {"lookback_hours": spec.lookback, "horizon_hours": spec.horizon,
                         "past_features": len(spec.past_features),
                         "future_features": len(spec.future_features),
                         "rnn_hidden": args.hidden, "train_minutes": round(train_minutes, 1)},
        "ensemble_weights": weights,
        "residual_quantiles": quantiles,
        "test": metrics,
    }
    paths["card"].write_text(json.dumps(card, indent=1))
    logger.info("Model card → {}", paths["card"])

    if args.promote_if_better:
        _champion_challenger(norm_store_raw=store, idx_te=idx_te, candidate_metrics=metrics,
                             device=device)
    return card


def _champion_challenger(norm_store_raw, idx_te, candidate_metrics: dict, device) -> bool:
    """Score the deployed champion on the candidate's test window; promote only if not worse."""
    cand = candidate_metrics["models"]["ensemble"]["aqi_mae"]
    try:
        spec = load_spec(CHECKPOINT_DIR)
        champ_store = normalise_store(norm_store_raw, spec)
        preds = predict_members(load_members(device, CHECKPOINT_DIR), champ_store, idx_te, spec, device)
        preds["ensemble"] = blend(preds, load_ensemble_weights(CHECKPOINT_DIR))
        champ = score(champ_store, idx_te, {"ensemble": preds["ensemble"]})["models"]["ensemble"]["aqi_mae"]
    except Exception as exc:
        logger.warning("No usable champion ({}); promoting candidate.", exc)
        champ = float("inf")

    promote = cand <= champ * PROMOTION_TOLERANCE
    logger.info("Champion AQI MAE={:.2f} | candidate={:.2f} → {}", champ, cand,
                "PROMOTE" if promote else "KEEP CHAMPION")
    if promote:
        for key, src in artifact_paths(CANDIDATE_DIR).items():
            if src.exists():
                shutil.copy2(src, artifact_paths(CHECKPOINT_DIR)[key])
    shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
    return promote


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the pan-India AQ forecast ensemble.")
    parser.add_argument("--source", choices=["parquet", "db"], default="parquet")
    parser.add_argument("--data", default="data/history.parquet")
    parser.add_argument("--days", type=int, default=120, help="History to load with --source db")
    parser.add_argument("--model", choices=["all", "lstm", "gru", "lgbm"], default="all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--stride", type=int, default=6, help="Hours between training windows per node")
    parser.add_argument("--max-train", type=int, default=None, help="Cap on NN training windows")
    parser.add_argument("--max-eval", type=int, default=20000, help="Cap on val/test windows")
    parser.add_argument("--lgbm-samples", type=int, default=100000)
    parser.add_argument("--val-days", type=int, default=21)
    parser.add_argument("--test-days", type=int, default=35)
    parser.add_argument("--promote-if-better", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_training(parse_args())
