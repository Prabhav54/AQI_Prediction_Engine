"""
run_pipeline.py
---------------
Single entry point for every batch job (GitHub Actions cron, Docker, local).

    python run_pipeline.py all          # ingest → forecast → score skill → prune
    python run_pipeline.py ingest       # ETL only
    python run_pipeline.py forecast     # ingest (no DB write of history) + forecast
    python run_pipeline.py skill        # score elapsed forecast runs
    python run_pipeline.py prune        # retention
    python run_pipeline.py backfill --start 2025-10-01 --end 2026-10-12 --out data/history.parquet
"""

import argparse
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import GRID_RESOLUTION_DEG, LOOKBACK_DAYS, OPEN_METEO_CALLS_PER_MIN
from logger import get_logger

logger = get_logger(__name__)


def cmd_ingest(args):
    from ingestion.pipeline import run_pan_india_ingestion

    return run_pan_india_ingestion(args.resolution, args.lookback_days)


def cmd_forecast(args, ingested=None):
    from database.db_client import register_model, write_forecasts
    from forecasting.ensemble import forecast_all_nodes, load_card

    nodes, frame = ingested or cmd_ingest(args)
    forecasts = forecast_all_nodes(nodes, frame)
    register_model(load_card())
    n = write_forecasts(forecasts)
    logger.info("Wrote {} forecast rows for {} nodes.", n, forecasts["location_hash"].nunique())
    return forecasts


def cmd_skill(_args):
    from database.db_client import evaluate_forecast_skill

    evaluate_forecast_skill()


def cmd_prune(_args):
    from database.db_client import prune

    prune()


def cmd_all(args):
    ingested = cmd_ingest(args)
    cmd_forecast(args, ingested)
    cmd_skill(args)
    cmd_prune(args)


def training_nodes(resolution: float, grid_stride: int) -> pd.DataFrame:
    """All city nodes + every `grid_stride`-th grid node (spatial spread for a global model)."""
    from ingestion.grid_generator import build_node_table

    nodes = build_node_table(resolution)
    grid = nodes[nodes["kind"] == "grid"].iloc[::grid_stride]
    return pd.concat([nodes[nodes["kind"] == "city"], grid], ignore_index=True)


def cmd_backfill(args):
    """
    Pull a long hourly history for training into parquet (resumable: one file
    per node batch; finished batches are skipped on re-run).
    """
    from ingestion.weather_client import fetch_batch

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    n_days = (end - start).days + 1
    nodes = training_nodes(args.resolution, args.grid_stride)

    # Size batches so one request fits inside the per-minute quota
    per_node = max(1.0, n_days / 14)
    batch = max(1, min(50, math.floor(OPEN_METEO_CALLS_PER_MIN / per_node)))
    n_batches = math.ceil(len(nodes) / batch)
    logger.info("Backfill {} → {} for {} nodes in {} batches of {} (~{:.0f} weighted calls)",
                start, end, len(nodes), n_batches, batch, 2 * len(nodes) * per_node)

    parts_dir = Path(args.out).with_suffix("").parent / f"{Path(args.out).stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_batches):
        part = parts_dir / f"{start}_{end}_{i:03d}.parquet"
        if part.exists():
            continue
        chunk = nodes.iloc[i * batch:(i + 1) * batch]
        df = fetch_batch(chunk, start_date=start, end_date=end)
        df = df.merge(chunk[["location_hash", "lat", "lon"]], on="location_hash")
        df.to_parquet(part, index=False)
        logger.info("Backfill batch {}/{} saved ({} rows)", i + 1, n_batches, len(df))

    full = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob(f"{start}_{end}_*.parquet"))])
    full.to_parquet(args.out, index=False)
    logger.info("History written → {} ({} rows, {} nodes)", args.out, len(full),
                full["location_hash"].nunique())

    if args.to_db:
        from database.db_client import refresh_aqi, upsert_nodes, write_raw_observations

        upsert_nodes(nodes)
        write_raw_observations(full)
        refresh_aqi(since=datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc))


def cmd_export_snapshot(args):
    """
    Dump what the Vercel API serves into one JSON file (web/public/snapshot.json).
    The dashboard falls back to it when no database is connected, so the
    deployed site always shows real model output.
    """
    import json

    from database.db_client import _read
    from forecasting.ensemble import load_card

    latest = _read("SELECT * FROM v_latest_aqi")
    fc = _read("""
        SELECT location_hash, forecast_generated_at, forecast_target_time, hours_ahead, aqi,
               aqi_lower, aqi_upper, aqi_category, prominent_pollutant, pm25, pm10, no2, so2, co, o3,
               aqi_lstm, aqi_gru, aqi_lgbm, model_version
        FROM v_latest_forecast ORDER BY location_hash, forecast_target_time
    """).round(2)
    hist = _read("""
        SELECT location_hash, timestamp, aqi FROM aqi_computed
        WHERE timestamp >= (SELECT MAX(timestamp) FROM aqi_computed) - INTERVAL '72 hours'
        ORDER BY location_hash, timestamp
    """)

    def recs(df):
        return json.loads(df.to_json(orient="records", date_format="iso"))

    compact = latest.rename(columns={"location_hash": "h", "location_name": "name", "timestamp": "t",
                                     "aqi_category": "cat", "prominent_pollutant": "pol"})
    fc_by = {h: g.drop(columns="location_hash") for h, g in fc.groupby("location_hash")}
    hist_by = {h: g.drop(columns="location_hash") for h, g in hist.groupby("location_hash")}
    card = load_card()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": recs(compact[["h", "name", "state", "lat", "lon", "kind", "t", "aqi", "cat", "pol"]]),
        "details": {
            row["location_hash"]: {
                "node": recs(latest[latest["location_hash"] == row["location_hash"]])[0],
                "forecast": recs(fc_by.get(row["location_hash"], fc.iloc[:0])),
                "history": recs(hist_by.get(row["location_hash"], hist.iloc[:0])),
            }
            for _, row in latest.iterrows()
        },
        "skill": {**{k: card.get(k) for k in ("model_version", "trained_at", "data", "test")},
                  "live_by_horizon": []},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, separators=(",", ":")))
    logger.info("Snapshot → {} ({} nodes, {:.1f} MB)", args.out, len(latest),
                Path(args.out).stat().st_size / 1e6)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pan-India AQ engine batch jobs")
    parser.add_argument("command", choices=["all", "ingest", "forecast", "skill", "prune",
                                            "backfill", "export-snapshot"])
    parser.add_argument("--resolution", type=float, default=GRID_RESOLUTION_DEG)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    # backfill
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    parser.add_argument("--start", default=(date.fromisoformat(yesterday) - timedelta(days=365)).isoformat())
    parser.add_argument("--end", default=yesterday)
    parser.add_argument("--grid-stride", type=int, default=10,
                        help="Backfill every Nth grid node besides all cities")
    parser.add_argument("--out", default="data/history.parquet")
    parser.add_argument("--to-db", action="store_true", help="Also load backfill into the DB")
    args = parser.parse_args(argv)

    {
        "all": cmd_all, "ingest": cmd_ingest, "forecast": cmd_forecast, "skill": cmd_skill,
        "prune": cmd_prune, "backfill": cmd_backfill, "export-snapshot": cmd_export_snapshot,
    }[args.command](args)


if __name__ == "__main__":
    main()
