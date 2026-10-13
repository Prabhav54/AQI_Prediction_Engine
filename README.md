# Pan-India AI Air Quality Forecasting Platform 🌍💨

[![CI](https://github.com/Prabhav54/AQI_Prediction_Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Prabhav54/AQI_Prediction_Engine/actions/workflows/ci.yml)
[![Pipeline](https://github.com/Prabhav54/AQI_Prediction_Engine/actions/workflows/pipeline.yml/badge.svg)](https://github.com/Prabhav54/AQI_Prediction_Engine/actions/workflows/pipeline.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%20%2B%20GRU-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB%20%2F%20Postgres-FDB515.svg?logo=postgresql)](https://www.timescale.com/)

**Live dashboard:** https://pan-india-aqi.vercel.app (snapshot mode until the Neon DB is attached; see Production below)

Forecasts the CPCB **Air Quality Index and all six regulated pollutants** (PM2.5, PM10, NO₂, SO₂, CO, O₃) **24 hours ahead, hour by hour, across the whole of India**. It covers 386 nodes: all 112 state capitals and million-plus cities, plus a 1° grid clipped to India's borders. A 0.5° grid (1,202 nodes) is a config switch.

A single **global** LSTM + GRU + LightGBM ensemble is trained across every node. It reads the last 168 hours at a node together with the next 24 hours of forecast weather. It predicts each pollutant's change from now, and CPCB AQI is then computed along the path of observed history plus forecast, exactly as the database computes it from observations.

---

## Results (held-out test, never seen in training or model selection)

Test window 2026-09-06 → 2026-10-12 (11,200 node × issue-time forecasts, 24 hourly steps each). The model was trained on 80 nodes covering Oct 2025 → Aug 2026 and is model version `2026.10.13-5f7e142`.

| Model | AQI MAE | AQI RMSE | Category hit | Skill vs persistence |
|---|---:|---:|---:|---:|
| Persistence | 39.0 | 57.1 | 54.7% | — |
| Seasonal-naive (same hour yesterday) | 16.6 | 29.2 | 78.4% | +57.5% |
| LSTM | 14.0 | 25.5 | 82.4% | +64.0% |
| GRU | 13.3 | 23.6 | 83.2% | +66.0% |
| LightGBM | 12.1 | 22.4 | 84.4% | +69.0% |
| **Ensemble** | **11.7** | **21.7** | **84.8%** | **+69.9%** |

| Pollutant (24h-ahead hourly) | PM2.5 | PM10 | NO₂ | SO₂ | CO | O₃ |
|---|---:|---:|---:|---:|---:|---:|
| Ensemble MAE (µg/m³; CO mg/m³) | 8.4 | 20.0 | 3.8 | 3.5 | 0.07 | 11.9 |
| Skill vs persistence | +35% | +25% | +68% | +37% | +51% | +75% |

* The ensemble beats its best single member (LightGBM) with Wilcoxon signed-rank p ≈ 1e-47.
* The 80% AQI prediction interval covers **86.8%** of test outcomes (the target is 80%).
* AQI MAE grows from 0.4 at +1h to 18.7 at +24h. Persistence reaches 60 by +15h.

* **Skill** = reduction in AQI MAE versus persistence ("the next 24h look like now"), averaged over every horizon from 1 to 24h and every node.
* **Category hit** = the share of forecasts that land in the correct CPCB band (Good … Severe).
* All figures come from `forecasting/checkpoints/model_card.json`, which every training run writes. Live skill of deployed forecasts is scored in-database every run (`forecast_skill` table) and shown on the dashboard.

---

## Architecture

```text
             ┌──────────────── GitHub Actions (every 3 h) ─────────────────┐
Open-Meteo   │  ingest            CPCB math (SQL)       ensemble inference │      Vercel
 weather  ──►│  386 nodes in ──►  refresh_aqi():   ──►  LSTM + GRU + LGBM  │──►  Neon Postgres ──► /api (FastAPI, read-only)
 CAMS AQ     │  8 batched calls   24h/8h windows,       all nodes in one   │                         + Leaflet dashboard
             │  (quota-throttled) sub-indices, AQI      batch, 80% bands   │
             └──────────────────────────────────────────────────────────────┘
             ┌──────────── GitHub Actions (weekly) ────────────┐
             │ backfill 180d → retrain → champion/challenger → │──► commits promoted checkpoints
             └─────────────────────────────────────────────────┘
```

| Layer | What it does | Where |
|---|---|---|
| **Ingestion** | Batched multi-location Open-Meteo requests (50 nodes per HTTP call). Weighted per-minute and per-hour quota throttle. `tenacity` exponential backoff on 429/5xx. A failed batch is skipped, not fatal. | `ingestion/weather_client.py`, `ingestion/pipeline.py` |
| **Node set** | India-clipped lattice generated once with GeoPandas and committed as CSV, so the cron job needs no GIS stack. City nodes override grid nodes within 25 km. | `ingestion/grid_generator.py`, `config/cities.py` |
| **Storage + CPCB math** | Portable schema: plain Postgres (Neon), upgraded automatically on TimescaleDB with hypertables, compression and a continuous aggregate. CPCB sub-indices and AQI are SQL functions. A Python mirror for forecast hours is tested against the SQL output. | `database/schema.sql` |
| **Forecasting** | Global multi-horizon ensemble: patched LSTM/GRU encoders with known future covariates (MQ-RNN style), plus a horizon-stacked LightGBM per pollutant. Per-pollutant blend weights are tuned on validation. | `forecasting/` |
| **Uncertainty** | 80% intervals from validation residual quantiles per horizon and pollutant (split-conformal), with test coverage reported. | `forecasting/evaluation.py` |
| **MLOps** | Hourly-class batch inference, in-database live skill scoring, a weekly retrain with a champion/challenger gate, and versioned model cards in a `model_registry` table. | `.github/workflows/`, `run_pipeline.py` |
| **Serving** | Vercel serverless API (two dependencies, edge-cached) and a Leaflet/Chart.js dashboard. If the DB is unreachable it falls back to a bundled snapshot. | `web/` |
| **Local stack** | FastAPI + Streamlit + TimescaleDB via Docker Compose. | `api/`, `ui/`, `docker-compose.yml` |

### Why these modelling choices

* **Predict the change, not the level.** Targets are `log1p(y[t+h]) − log1p(y[t])`, scaled per horizon. Persistence becomes the zero baseline, and the log scale tames dust-storm and stubble-burning spikes. The loss is Huber.
* **Use forecast weather.** Tomorrow's PM depends mostly on tomorrow's boundary-layer height, wind and rain. The old model held weather constant for 24h; this one feeds the next-24h weather forecast directly into the decoder.
* **Direct multi-horizon, not autoregressive.** All 24 steps come from one forward pass, so errors don't compound.
* **Forecast pollutants, then derive AQI.** AQI is a 24h/8h rolling mean, so the CPCB math is applied to observed history plus forecast. The forecast AQI is therefore exactly the number CPCB will publish when those hours are observed.
* **One global model.** Training on ~140 nodes across every climate zone lets the model forecast rural grid nodes that have no special history.
* **Patching.** Every 4 hours are concatenated into one encoder step (PatchTST). The RNN still sees all 168 hours but trains about 50× faster on CPU, which is what makes weekly retraining in free CI possible.

---

## Repository layout

```text
.github/workflows/  ci.yml · pipeline.yml (3-hourly) · retrain.yml (weekly)
api/                FastAPI backend for local / Docker use
config/             settings.py · cities.py (112 city nodes)
database/           schema.sql (tables, CPCB SQL functions, views) · setup_db.py · db_client.py
forecasting/        dataset.py · model.py · ensemble.py · train.py · evaluation.py · checkpoints/
ingestion/          weather_client.py · pipeline.py · grid_generator.py · nodes/*.csv · geocoder.py · gee_client.py
proxy_model/        XGBoost AOD→PM gap-filler (only fills hours where CAMS PM is missing)
ui/app.py           Streamlit dashboard
web/                Vercel deployment: api/index.py (FastAPI) + public/index.html, public/snapshot.json
run_pipeline.py     all | ingest | forecast | skill | prune | backfill | export-snapshot
```

---

## Running it

### Local (conda / venv)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Postgres or TimescaleDB on localhost (or: docker compose up -d db)
export DATABASE_URL_SYNC=postgresql://aq_user:aq_pass@localhost:5432/air_quality_db
python database/setup_db.py              # schema + 386 nodes (idempotent)

python run_pipeline.py all               # ingest → forecast → skill → prune
uvicorn api.main:app --reload            # http://localhost:8000/docs
streamlit run ui/app.py                  # http://localhost:8501
```

### Docker

```bash
docker compose up -d                     # TimescaleDB + schema + API + Streamlit
docker compose run --rm pipeline         # one pipeline pass
```

### Retraining

```bash
python run_pipeline.py backfill --start 2025-10-01 --end 2026-10-12 --out data/history.parquet
python forecasting/train.py --source parquet --data data/history.parquet            # writes checkpoints + model card
python forecasting/train.py --source parquet --data data/history.parquet --promote-if-better
```

### Production (GitHub Actions + Neon + Vercel)

1. In the Vercel project (root directory `web`), add the **Neon** integration from the Marketplace. This injects `DATABASE_URL`.
2. Copy that connection string into the GitHub repo secret **`DATABASE_URL`**.
3. Run the **Pan-India pipeline** workflow once with **setup_db = true**. After that, the cron keeps it fresh every 3 hours.

The cadence is deliberate. CAMS updates twice a day, and 386 nodes × 2 endpoints × 8 runs ≈ 6.2k calls/day fits Open-Meteo's free 10k/day quota.

---

## Data sources and honest caveats

* **Pollutants:** Copernicus CAMS global atmospheric composition forecasts (~45 km) via the Open-Meteo Air Quality API. **These are model fields, not station measurements.** The forecaster learns to predict CAMS' future values, so its error is relative to CAMS and not to CPCB monitors.
* **CAMS ozone is biased high over coastal and peninsular India.** For example, Mumbai's 8-hour O₃ often exceeds 200 µg/m³ in CAMS, far above typical CPCB station readings. This frequently makes O₃ the "prominent pollutant". The dashboard shows the prominent pollutant so this stays visible. Fixing it properly needs station data (see the roadmap).
* **Weather:** the Open-Meteo forecast API for live runs and the historical-forecast archive for training. Training uses archived short-lead forecasts as "future weather", which are slightly better than true 24h-lead forecasts, so live skill can be a little lower than offline skill. `forecast_skill` measures exactly that gap.
* **Google Earth Engine** (`ingestion/gee_client.py`, Sentinel-5P / MODIS) remains as an optional research path. Satellite overpasses are daily and cloud-gapped, so the hourly pipeline uses Open-Meteo.

## Roadmap

* Bias-correct against CPCB / OpenAQ station observations, so that CAMS becomes a feature and a baseline instead of the target.
* A Temporal Fusion Transformer member with native quantile outputs.
* Rolling-origin backtests across all seasons (Diwali, stubble burning, monsoon) instead of a single test window.
