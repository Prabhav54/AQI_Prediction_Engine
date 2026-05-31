# Pan-India Geospatial Air Quality Ingestion & Forecasting Engine

A modular MLOps pipeline that fuses satellite imagery (Sentinel-5P, MODIS),
meteorological reanalysis (Open-Meteo), and deep-learning forecasting to
compute real-time CPCB AQI for any Indian city.

---

## Python Version — Why 3.11?

**Use Python 3.11.x. Do not use 3.12 or 3.13.**

| Reason | Detail |
|--------|--------|
| **PyTorch stability** | PyTorch 2.x wheel builds are most battle-tested on 3.11; 3.12 support was added later and still has edge-case issues with `torch.compile` |
| **earthengine-api** | The GEE Python client has had intermittent import failures on 3.12 due to `distutils` removal |
| **TimescaleDB / asyncpg** | `asyncpg` binary wheels for 3.12 on Linux ARM (e.g. Raspberry Pi / cloud ARM VMs) lag behind 3.11 |
| **Ecosystem maturity** | 3.11 ships with significant CPython speed-ups (+10–60% vs 3.10) while being fully stable for all our dependencies |

---

## Environment Setup (Anaconda)

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/pan-india-aq-engine.git
cd pan-india-aq-engine
```

### Step 2 — Create the Conda environment

```bash
# Creates a new env named 'aq_engine' with Python 3.11 + all dependencies
conda env create -f environment.yml
```

This typically takes 3–5 minutes on first run (downloading packages).

### Step 3 — Activate the environment

```bash
conda activate aq_engine
```

You should see `(aq_engine)` in your shell prompt. **All subsequent commands
must be run with this environment active.**

### Step 4 — Install the project as an editable package

```bash
# This lets all modules import each other without sys.path hacks
pip install -e .
```

### Step 5 — Configure secrets

```bash
cp .env.example .env
# Open .env and fill in your GEE service account + DB credentials
```

### Step 6 — Verify the setup

```bash
# Run the test suite (no GEE credentials needed — uses mock satellite data)
pytest tests/ -v

# Run the Module 1 pipeline against a real location (mock satellite mode)
aq-ingest "Kolkata, West Bengal" --mock-satellite --days 3
```

---

## Updating the Environment

If `environment.yml` changes (e.g. new dependency added):

```bash
conda env update -f environment.yml --prune
# --prune removes packages that are no longer listed
```

---

## Removing the Environment

```bash
conda deactivate
conda env remove -n aq_engine
```

---

## Google Earth Engine Setup

1. Sign up at [earthengine.google.com](https://earthengine.google.com/)
2. Create a GCP project and enable the Earth Engine API
3. Create a Service Account, grant it the **Earth Engine Resource Viewer** role
4. Download the JSON key → save as `config/gee_key.json`
5. Set `GEE_SERVICE_ACCOUNT` and `GEE_KEY_FILE` in your `.env`

For local interactive development (no service account needed):
```bash
python -c "import ee; ee.Authenticate(); ee.Initialize()"
# Opens a browser for OAuth login — credentials cached in ~/.config/earthengine/
```

---

## Project Structure

```
pan_india_aq_engine/
├── config/settings.py          # All constants & env variable loading
├── ingestion/
│   ├── geocoder.py             # Module 1A: Nominatim geocoding
│   ├── gee_client.py           # Module 1B: GEE satellite pull
│   ├── weather_client.py       # Module 1C: Open-Meteo weather pull
│   └── pipeline.py             # Module 1 orchestrator
├── proxy_model/
│   ├── train.py                # Module 2: XGBoost AOD→PM2.5 training
│   └── predict.py              # Module 2: inference wrapper
├── database/
│   ├── schema.sql              # TimescaleDB hypertable definitions
│   ├── aqi_sql.sql             # Module 3: CPCB sub-index window functions
│   └── db_client.py            # SQLAlchemy async helpers
├── forecasting/
│   ├── dataset.py              # Module 4: sliding-window sequence builder
│   ├── model.py                # Module 4: PyTorch LSTM
│   └── train.py                # Module 4: training loop
├── api/
│   ├── main.py                 # Module 5: FastAPI app
│   └── routes/
│       ├── ingest.py           # POST /ingest
│       └── forecast.py         # GET /forecast
├── ui/app.py                   # Module 5: Streamlit frontend
├── exceptions.py               # Project-wide exception hierarchy
├── logger.py                   # Loguru-based centralised logging
├── utils.py                    # CPCB AQI calc, validators, helpers
├── setup.py                    # Editable package install
├── environment.yml             # Conda environment spec (Python 3.11)
├── requirements.txt            # pip fallback
├── .env.example                # Secret template
└── .gitignore
```

---

## Module Roadmap

| Module | Status | Description |
|--------|--------|-------------|
| 1 — Ingestion | ✅ Complete | Geocoding, GEE satellite, Open-Meteo weather |
| 2 — Proxy Model | 🔜 Next | XGBoost AOD → PM2.5/PM10 regression |
| 3 — Database | 🔜 | TimescaleDB hypertables + CPCB SQL |
| 4 — Forecasting | 🔜 | PyTorch LSTM 24-hour AQI forecast |
| 5 — API + UI | 🔜 | FastAPI + Streamlit |

---

## License

MIT License