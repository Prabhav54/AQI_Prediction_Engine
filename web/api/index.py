"""
web/api/index.py
----------------
Vercel serverless API for the dashboard (read-only).

All heavy lifting — ingestion, CPCB math, the LSTM + GRU + LightGBM
ensemble — happens in the batch pipeline (GitHub Actions). This function
only reads the results from Postgres (Neon), so it ships with two
dependencies and answers in one indexed query. Responses are cached at
Vercel's edge until the next pipeline run could have changed them.
"""

import json
import os
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row

DB_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
CACHE = "public, s-maxage=600, stale-while-revalidate=3600"

app = FastAPI(title="Pan-India AQI API", docs_url="/api/docs", openapi_url="/api/openapi.json")


def _default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(type(o))


def _json(payload, cache: str = CACHE) -> JSONResponse:
    body = json.loads(json.dumps(payload, default=_default))
    return JSONResponse(body, headers={"Cache-Control": cache})


@contextmanager
def _conn():
    if not DB_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured.")
    with psycopg.connect(DB_URL, row_factory=dict_row, connect_timeout=10) as conn:
        yield conn


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with _conn() as conn:
        return conn.execute(sql, params).fetchall()


@app.get("/api/health")
def health():
    try:
        n = _rows("SELECT COUNT(*) AS n FROM grid_nodes")[0]["n"]
        latest = _rows("SELECT MAX(forecast_generated_at) AS t FROM aqi_forecasts")[0]["t"]
        return _json({"db": "ok", "nodes": n, "latest_forecast_run": latest}, cache="no-store")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"db error: {exc}")


@app.get("/api/nodes")
def nodes():
    """Latest AQI at every node — compact rows for the map."""
    rows = _rows("""
        SELECT location_hash AS h, location_name AS name, state, lat, lon, kind,
               timestamp AS t, aqi, aqi_category AS cat, prominent_pollutant AS pol
        FROM v_latest_aqi
    """)
    return _json({"generated_at": datetime.utcnow(), "nodes": rows})


@app.get("/api/node/{location_hash}")
def node(location_hash: str):
    """Everything the detail panel needs for one node."""
    with _conn() as conn:
        info = conn.execute("SELECT * FROM v_latest_aqi WHERE location_hash = %s",
                            (location_hash,)).fetchone()
        if info is None:
            raise HTTPException(status_code=404, detail="Unknown node or no recent data.")
        forecast = conn.execute("""
            SELECT forecast_generated_at, forecast_target_time, hours_ahead, aqi, aqi_lower, aqi_upper,
                   aqi_category, prominent_pollutant, pm25, pm10, no2, so2, co, o3,
                   aqi_lstm, aqi_gru, aqi_lgbm, model_version
            FROM v_latest_forecast WHERE location_hash = %s ORDER BY forecast_target_time
        """, (location_hash,)).fetchall()
        history = conn.execute("""
            SELECT timestamp, aqi FROM aqi_computed
            WHERE location_hash = %s AND timestamp >= NOW() - INTERVAL '72 hours'
            ORDER BY timestamp
        """, (location_hash,)).fetchall()
    return _json({"node": info, "forecast": forecast, "history": history})


@app.get("/api/nearest")
def nearest(lat: float = Query(..., ge=6, le=38), lon: float = Query(..., ge=68, le=98)):
    rows = _rows("""
        SELECT location_hash, location_name, lat, lon,
               111.32 * SQRT(POWER(lat - %s, 2) + POWER((lon - %s) * COS(RADIANS(%s)), 2)) AS km
        FROM grid_nodes ORDER BY km LIMIT 1
    """, (lat, lon, lat))
    if not rows or rows[0]["km"] > 150:
        raise HTTPException(status_code=404, detail="No node within 150 km.")
    return _json(rows[0])


@app.get("/api/skill")
def skill(days: int = Query(7, ge=1, le=30)):
    """Offline test metrics of the deployed model + live skill of recent forecast runs."""
    with _conn() as conn:
        card = conn.execute("""
            SELECT card FROM model_registry ORDER BY last_used_at DESC LIMIT 1
        """).fetchone()
        live = conn.execute("""
            SELECT hours_ahead,
                   SUM(n_nodes) AS n,
                   SUM(aqi_mae * n_nodes) / NULLIF(SUM(n_nodes), 0) AS aqi_mae,
                   SUM(persistence_aqi_mae * n_nodes) / NULLIF(SUM(n_nodes), 0) AS persistence_aqi_mae,
                   SUM(category_accuracy * n_nodes) / NULLIF(SUM(n_nodes), 0) AS category_accuracy
            FROM forecast_skill
            WHERE forecast_generated_at >= NOW() - make_interval(days => %s)
            GROUP BY hours_ahead ORDER BY hours_ahead
        """, (days,)).fetchall()
    card = (card or {}).get("card") or {}
    return _json({
        "model_version": card.get("model_version"),
        "trained_at": card.get("trained_at"),
        "data": card.get("data"),
        "test": card.get("test"),
        "live_by_horizon": live,
    })
