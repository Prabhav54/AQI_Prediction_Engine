"""
forecasting/evaluation.py
--------------------------
Held-out evaluation of the ensemble against its members and naive baselines.

Every number in the model card / README comes from here:
  - Baselines: persistence ("next 24h = now") and seasonal-naive ("same hour yesterday")
  - Per pollutant: MAE / RMSE in concentration units, skill vs persistence
  - CPCB AQI (computed along history + forecast): MAE, RMSE, category hit-rate,
    share within ±20 AQI, MAE per horizon
  - Wilcoxon signed-rank test: ensemble vs best single member (per-sample AQI errors)
  - Empirical coverage of the 80% interval

Run standalone (re-scores the deployed checkpoints on the latest test window):
    python forecasting/evaluation.py --data data/history.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import POLLUTANTS
from forecasting.dataset import HORIZON, history_conc, last_log, true_deltas
from logger import get_logger
from utils import CPCB_INDEX_LEVELS, aqi_along_paths

logger = get_logger(__name__)

ACCURACY_TOLERANCE = 20   # "within one sub-band" — kept for comparability with the old metric


def _category_idx(aqi: np.ndarray) -> np.ndarray:
    return np.searchsorted(CPCB_INDEX_LEVELS[1:-1], np.round(aqi), side="left")


def seasonal_naive_deltas(store, index) -> np.ndarray:
    """Δ for 'same hour yesterday': y[t+h-24] - y[t]."""
    return np.stack([
        store[n].ylog[t + 1 - 24:t + HORIZON + 1 - 24] - store[n].ylog[t] for n, t in index
    ])


def score(store, index, candidates: dict[str, np.ndarray], interval: dict | None = None) -> dict:
    """
    candidates : {name: (N, 24, 6) log-deltas}. Baselines are added automatically.
    interval   : optional {"lo": deltas, "hi": deltas} for the ensemble.
    """
    from forecasting.ensemble import deltas_to_conc

    last = last_log(store, index)
    hist = history_conc(store, index)
    true_d = true_deltas(store, index)
    y_conc = deltas_to_conc(last, true_d)
    y_aqi, _ = aqi_along_paths(hist, y_conc)

    cands = {
        "persistence": np.zeros_like(true_d),
        "seasonal_naive": seasonal_naive_deltas(store, index),
        **candidates,
    }

    results, abs_err = {}, {}
    for name, d in cands.items():
        conc = deltas_to_conc(last, d)
        aqi, _ = aqi_along_paths(hist, conc)
        err = aqi - y_aqi
        abs_err[name] = np.abs(err).mean(axis=1)            # per-sample mean over horizons
        results[name] = {
            "aqi_mae": float(np.abs(err).mean()),
            "aqi_rmse": float(np.sqrt((err ** 2).mean())),
            "aqi_bias": float(err.mean()),
            "aqi_category_accuracy": float((_category_idx(aqi) == _category_idx(y_aqi)).mean()),
            f"aqi_within_{ACCURACY_TOLERANCE}": float((np.abs(err) <= ACCURACY_TOLERANCE).mean()),
            "aqi_mae_by_horizon": np.abs(err).mean(axis=0).round(2).tolist(),
            "pollutants": {
                p: {
                    "mae": float(np.abs(conc[..., j] - y_conc[..., j]).mean()),
                    "rmse": float(np.sqrt(((conc[..., j] - y_conc[..., j]) ** 2).mean())),
                }
                for j, p in enumerate(POLLUTANTS)
            },
        }

    base = results["persistence"]
    for name, r in results.items():
        r["aqi_skill_vs_persistence"] = 1 - r["aqi_mae"] / base["aqi_mae"]
        for p in POLLUTANTS:
            r["pollutants"][p]["skill_vs_persistence"] = (
                1 - r["pollutants"][p]["mae"] / base["pollutants"][p]["mae"]
            )

    out = {"n_samples": int(len(index)), "models": results}

    members = [m for m in ("lstm", "gru", "lgbm") if m in results]
    if "ensemble" in results and members:
        best = min(members, key=lambda m: results[m]["aqi_mae"])
        _, pval = stats.wilcoxon(abs_err["ensemble"], abs_err[best], alternative="less")
        out["wilcoxon_ensemble_vs_best_member"] = {"best_member": best, "p_value": float(pval)}

    if interval is not None:
        lo, _ = aqi_along_paths(hist, deltas_to_conc(last, interval["lo"]))
        hi, _ = aqi_along_paths(hist, deltas_to_conc(last, interval["hi"]))
        out["interval_80_coverage_aqi"] = float(((y_aqi >= lo) & (y_aqi <= hi)).mean())

    for name in cands:
        r = results[name]
        logger.info("{:<15} AQI MAE={:6.2f}  RMSE={:6.2f}  cat-acc={:5.1%}  skill={:+.1%}",
                    name, r["aqi_mae"], r["aqi_rmse"], r["aqi_category_accuracy"],
                    r["aqi_skill_vs_persistence"])
    return out


def residual_quantiles(true_d: np.ndarray, pred_d: np.ndarray, lo: float = 0.10, hi: float = 0.90) -> dict:
    """Split-conformal style: per (horizon, pollutant) quantiles of validation residuals."""
    r = true_d - pred_d
    return {"q10": np.quantile(r, lo, axis=0).tolist(), "q90": np.quantile(r, hi, axis=0).tolist()}


if __name__ == "__main__":
    import torch

    from forecasting.dataset import build_store, chronological_splits, normalise_store, sample_index
    from forecasting.ensemble import blend, load_card, load_ensemble_weights, load_members, predict_members
    from forecasting.model import load_spec
    from forecasting.train import load_frame

    parser = argparse.ArgumentParser(description="Re-score deployed checkpoints on the latest test window.")
    parser.add_argument("--data", default="data/history.parquet")
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()

    nodes, frame = load_frame("parquet", args.data, days=0)
    store = build_store(frame, nodes)
    *_, (te0, _) = chronological_splits(store)
    idx = sample_index(store, args.stride, te0)
    spec = load_spec()
    device = torch.device("cpu")
    preds = predict_members(load_members(device), normalise_store(store, spec), idx, spec, device)
    preds["ensemble"] = blend(preds, load_ensemble_weights())
    q = load_card().get("residual_quantiles")
    interval = None if not q else {"lo": preds["ensemble"] + np.asarray(q["q10"]),
                                   "hi": preds["ensemble"] + np.asarray(q["q90"])}
    print(json.dumps(score(store, idx, preds, interval), indent=1)[:4000])
