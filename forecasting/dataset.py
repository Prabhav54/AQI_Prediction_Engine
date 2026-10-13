"""
forecasting/dataset.py
----------------------
Module 4 — Feature engineering + windowing for the global pan-India forecaster.

One model is trained across every node (not one model per city), so it
learns how pollution responds to weather everywhere and can forecast any
grid node, including ones with little history.

Per sample (node n, issue hour t):
  past   : 168h × PAST_FEATURES    — pollutants (log), weather, AOD, time, location
  future :  24h × FUTURE_FEATURES  — *forecast* weather + calendar for t+1..t+24
  target :  24h × 6 pollutants     — log1p(y[t+h]) - log1p(y[t]), scaled per (h, pollutant)

Predicting the *change* from the last observed value makes persistence
the zero baseline: the models only have to learn what moves the air.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.settings import FORECAST_HOURS, LSTM_LOOKBACK_HOURS, POLLUTANTS
from exceptions import SequenceTooShortError

LOOKBACK = LSTM_LOOKBACK_HOURS   # 168
HORIZON = FORECAST_HOURS         # 24

LOG_COLS = POLLUTANTS + ["dust"]
WEATHER_FEATURES = [
    "temp_c", "humidity_pct", "wind_u", "wind_v", "wind_speed_ms",
    "precip_log", "pressure_hpa", "blh_log", "radiation_wm2",
]
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
PAST_FEATURES = (
    [f"log_{c}" for c in LOG_COLS] + ["aod"] + WEATHER_FEATURES + TIME_FEATURES + ["lat_norm", "lon_norm"]
)
FUTURE_FEATURES = WEATHER_FEATURES + TIME_FEATURES

# Backwards-compatible names used elsewhere in the codebase
LSTM_FEATURE_COLS = PAST_FEATURES
N_FEATURES = len(PAST_FEATURES)

IST = "Asia/Kolkata"


# ===========================================================================
# Feature engineering (shared by training and inference)
# ===========================================================================

def engineer_features(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """
    One node's hourly frame (DatetimeIndex, UTC) → model features.
    Reindexes to a gap-free hourly axis; gaps longer than 3h stay NaN
    and are excluded later by the window validity mask.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full).ffill(limit=3)

    out = pd.DataFrame(index=df.index)
    for c in LOG_COLS:
        out[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    out["aod"] = df["aod"]

    rad = np.deg2rad(df["wind_dir_deg"])
    out["temp_c"] = df["temp_c"]
    out["humidity_pct"] = df["humidity_pct"]
    out["wind_u"] = -df["wind_speed_ms"] * np.sin(rad)   # meteorological → vector components
    out["wind_v"] = -df["wind_speed_ms"] * np.cos(rad)
    out["wind_speed_ms"] = df["wind_speed_ms"]
    out["precip_log"] = np.log1p(df["precip_mm"].clip(lower=0))
    out["pressure_hpa"] = df["pressure_hpa"]
    out["blh_log"] = np.log1p(df["boundary_layer_m"].clip(lower=0))
    out["radiation_wm2"] = df["radiation_wm2"]

    # Calendar in local time — traffic and cooking peaks follow IST, not UTC
    local = df.index.tz_convert(IST)
    hour, dow, doy = local.hour.values, local.dayofweek.values, local.dayofyear.values
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    out["is_weekend"] = (dow >= 5).astype(float)

    # Static location (India bounding box → [0, 1])
    out["lat_norm"] = (lat - 6.5) / (37.5 - 6.5)
    out["lon_norm"] = (lon - 68.0) / (97.5 - 68.0)
    return out


# ===========================================================================
# Per-node arrays
# ===========================================================================

@dataclass
class NodeSeries:
    location_hash: str
    times: pd.DatetimeIndex
    past: np.ndarray      # (T, F_past)   raw features
    future: np.ndarray    # (T, F_future) raw covariates (indexed at the *target* hour)
    ylog: np.ndarray      # (T, 6)        log1p pollutant concentrations
    valid: np.ndarray = field(default=None)   # (T,) bool — t is a usable issue hour


def _window_ok(bad: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where bad[lo:hi] (half-open) contains no bad rows."""
    c = np.concatenate([[0], np.cumsum(bad)])
    return (c[hi] - c[lo]) == 0


def build_node_series(df: pd.DataFrame, location_hash: str, lat: float, lon: float,
                      require_targets: bool = True) -> NodeSeries:
    feats = engineer_features(df, lat, lon)
    past = feats[PAST_FEATURES].to_numpy(np.float32)
    fut = feats[FUTURE_FEATURES].to_numpy(np.float32)
    ylog = feats[[f"log_{p}" for p in POLLUTANTS]].to_numpy(np.float32)

    T = len(feats)
    t = np.arange(T)
    valid = (t >= LOOKBACK - 1) & (t + HORIZON < T if require_targets else t + HORIZON <= T - 1)
    tt = np.clip(t, LOOKBACK - 1, max(T - HORIZON - 1, LOOKBACK - 1))
    valid &= _window_ok(~np.isfinite(past).all(1), tt - LOOKBACK + 1, tt + 1)
    valid &= _window_ok(~np.isfinite(fut).all(1), tt + 1, np.minimum(tt + HORIZON + 1, T))
    if require_targets:
        valid &= _window_ok(~np.isfinite(ylog).all(1), tt, np.minimum(tt + HORIZON + 1, T))
    return NodeSeries(location_hash, feats.index, past, fut, ylog, valid)


def build_store(frame: pd.DataFrame, nodes: pd.DataFrame, require_targets: bool = True) -> list[NodeSeries]:
    """Long frame (location_hash, timestamp, raw cols) → list of NodeSeries."""
    coords = nodes.set_index("location_hash")[["lat", "lon"]]
    store = []
    for h, g in frame.groupby("location_hash"):
        if h not in coords.index:
            continue
        g = g.set_index(pd.DatetimeIndex(pd.to_datetime(g["timestamp"], utc=True)))
        if len(g) < LOOKBACK + 1:
            continue
        lat, lon = coords.loc[h]
        store.append(build_node_series(g, h, float(lat), float(lon), require_targets))
    if not store:
        raise SequenceTooShortError(
            f"No node has the {LOOKBACK}h of history needed for a forecast window."
        )
    return store


# ===========================================================================
# Scaling (fit on the training split only — stored inside every checkpoint)
# ===========================================================================

@dataclass
class FeatureSpec:
    past_mean: np.ndarray
    past_std: np.ndarray
    fut_mean: np.ndarray
    fut_std: np.ndarray
    target_std: np.ndarray            # (HORIZON, 6) std of log-deltas per horizon & pollutant
    past_features: list = field(default_factory=lambda: list(PAST_FEATURES))
    future_features: list = field(default_factory=lambda: list(FUTURE_FEATURES))
    pollutants: list = field(default_factory=lambda: list(POLLUTANTS))
    lookback: int = LOOKBACK
    horizon: int = HORIZON

    def to_json(self, path: Path) -> None:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        Path(path).write_text(json.dumps(d, indent=1))

    @classmethod
    def from_json(cls, path: Path) -> "FeatureSpec":
        d = json.loads(Path(path).read_text())
        for k in ("past_mean", "past_std", "fut_mean", "fut_std", "target_std"):
            d[k] = np.asarray(d[k], dtype=np.float32)
        return cls(**d)


def fit_spec(store: list[NodeSeries], index: np.ndarray) -> FeatureSpec:
    """Fit scalers on the hours covered by the training index."""
    rows_p, rows_f, deltas = [], [], []
    rng = np.random.default_rng(0)
    pick = index[rng.choice(len(index), size=min(len(index), 20000), replace=False)]
    for n, t in pick:
        s = store[n]
        rows_p.append(s.past[t])
        rows_f.append(s.future[t + 1])
        deltas.append(s.ylog[t + 1:t + HORIZON + 1] - s.ylog[t])
    P, Fu, D = np.stack(rows_p), np.stack(rows_f), np.stack(deltas)

    def _ms(a):
        m, sd = np.nanmean(a, 0), np.nanstd(a, 0)
        return m.astype(np.float32), np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    pm, ps = _ms(P)
    fm, fs = _ms(Fu)
    ts = np.nanstd(D, 0)
    return FeatureSpec(pm, ps, fm, fs, np.where(ts < 1e-6, 1.0, ts).astype(np.float32))


def normalise_store(store: list[NodeSeries], spec: FeatureSpec) -> list[NodeSeries]:
    """Return copies with past/future standardised (NaNs → 0 after scaling)."""
    out = []
    for s in store:
        p = np.nan_to_num((s.past - spec.past_mean) / spec.past_std).astype(np.float32)
        f = np.nan_to_num((s.future - spec.fut_mean) / spec.fut_std).astype(np.float32)
        out.append(NodeSeries(s.location_hash, s.times, p, f, s.ylog, s.valid))
    return out


# ===========================================================================
# Sample index + splits
# ===========================================================================

def sample_index(store: list[NodeSeries], stride: int = 1,
                 start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> np.ndarray:
    """(node_idx, t) pairs for valid issue hours in [start, end), every `stride` hours."""
    idx = []
    for n, s in enumerate(store):
        mask = s.valid.copy()
        if start is not None:
            mask &= s.times >= start
        if end is not None:
            mask &= s.times < end
        ts = np.flatnonzero(mask)
        ts = ts[ts % stride == 0]
        idx.append(np.column_stack([np.full(len(ts), n), ts]))
    return np.concatenate(idx).astype(np.int64) if idx else np.empty((0, 2), np.int64)


def chronological_splits(store: list[NodeSeries], val_days: int = 21, test_days: int = 35):
    """Train | val | test by issue time, with a HORIZON gap so no target leaks across splits."""
    t_max = max(s.times[s.valid].max() for s in store if s.valid.any())
    test_start = t_max - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    gap = pd.Timedelta(hours=HORIZON)
    return (None, val_start - gap), (val_start, test_start - gap), (test_start, None)


# ===========================================================================
# PyTorch dataset
# ===========================================================================

class AQISequenceDataset(Dataset):
    """Windows are sliced on the fly — no (N, 168, F) tensor is ever materialised."""

    def __init__(self, store: list[NodeSeries], index: np.ndarray, spec: FeatureSpec):
        self.store, self.index, self.spec = store, index, spec

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        n, t = self.index[i]
        s = self.store[n]
        past = s.past[t - LOOKBACK + 1:t + 1]
        fut = s.future[t + 1:t + HORIZON + 1]
        y = (s.ylog[t + 1:t + HORIZON + 1] - s.ylog[t]) / self.spec.target_std
        return torch.from_numpy(past), torch.from_numpy(fut), torch.from_numpy(np.nan_to_num(y))


def nn_batch(store: list[NodeSeries], index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack past/future windows for a set of samples (inference / evaluation)."""
    past = np.stack([store[n].past[t - LOOKBACK + 1:t + 1] for n, t in index])
    fut = np.stack([store[n].future[t + 1:t + HORIZON + 1] for n, t in index])
    return past, fut


def true_deltas(store: list[NodeSeries], index: np.ndarray) -> np.ndarray:
    """(N, HORIZON, 6) unscaled log-deltas."""
    return np.stack([store[n].ylog[t + 1:t + HORIZON + 1] - store[n].ylog[t] for n, t in index])


def last_log(store: list[NodeSeries], index: np.ndarray) -> np.ndarray:
    """(N, 6) log1p concentrations at the issue hour."""
    return np.stack([store[n].ylog[t] for n, t in index])


def history_conc(store: list[NodeSeries], index: np.ndarray, hours: int = 23) -> np.ndarray:
    """(N, hours, 6) concentrations for the hours before t+1 (t-hours+1 .. t), for CPCB windows."""
    return np.stack([np.expm1(store[n].ylog[t - hours + 1:t + 1]) for n, t in index])


# ===========================================================================
# LightGBM tabular features
# ===========================================================================

_LAGS = (1, 3, 6, 24)
_ROLLS = (24, 168)


def lgbm_features(store: list[NodeSeries], index: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """
    Horizon-stacked tabular matrix: one row per (sample, h).

    Columns: current past features, pollutant lags & rolling means (as
    deltas from now), same-hour-yesterday for the target hour, forecast
    weather/calendar at t+h, and h itself.
    """
    rows = []
    for (n, t), h in zip(index, horizons):
        s = store[n]
        y = s.ylog
        cur = y[t]
        lag = [y[t - k] - cur for k in _LAGS]
        roll = [np.nanmean(y[t - w + 1:t + 1], 0) - cur for w in _ROLLS]
        seasonal = y[t + h - 24] - cur          # same hour, previous day (h ≤ 24 → already observed)
        rows.append(np.concatenate([s.past[t], *lag, *roll, seasonal, s.future[t + h], [h]]))
    return np.asarray(rows, dtype=np.float32)


def lgbm_feature_names() -> list[str]:
    names = list(PAST_FEATURES)
    for k in _LAGS:
        names += [f"d_lag{k}_{p}" for p in POLLUTANTS]
    for w in _ROLLS:
        names += [f"d_roll{w}_{p}" for p in POLLUTANTS]
    names += [f"d_yday_{p}" for p in POLLUTANTS]
    names += [f"fut_{c}" for c in FUTURE_FEATURES] + ["horizon"]
    return names
