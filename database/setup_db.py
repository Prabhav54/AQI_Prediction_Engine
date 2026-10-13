"""
database/setup_db.py
--------------------
Apply database/schema.sql to whatever DATABASE_URL_SYNC points at —
local Postgres/TimescaleDB, the docker-compose `db` service, or Neon.

The schema is idempotent, so this is safe to run on every deploy.

Usage
-----
  python database/setup_db.py             # apply schema + register nodes
  python database/setup_db.py --reset     # DROP all project tables first (destructive)
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import GRID_RESOLUTION_DEG
from database.db_client import get_sync_engine, upsert_nodes
from ingestion.grid_generator import build_node_table
from logger import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_DROP_SQL = """
DO $$ BEGIN
    -- aqi_daily is a continuous aggregate on Timescale but a plain view elsewhere
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'aqi_daily') THEN
        DROP MATERIALIZED VIEW aqi_daily CASCADE;
    END IF;
END $$;
DROP VIEW IF EXISTS aqi_daily, v_latest_aqi, v_latest_forecast CASCADE;
DROP TABLE IF EXISTS model_registry, forecast_skill, aqi_forecasts, aqi_computed, raw_observations, grid_nodes CASCADE;
"""


def apply_schema(reset: bool = False) -> None:
    engine = get_sync_engine()
    # Raw DBAPI connection: schema.sql contains $$-quoted function bodies and
    # DO blocks that SQLAlchemy's text() parameter parsing would trip over.
    fairy = engine.raw_connection()
    raw = fairy.dbapi_connection  # the real psycopg2 connection, not the pool proxy
    try:
        raw.autocommit = True
        cur = raw.cursor()
        if reset:
            logger.warning("Dropping all project tables (--reset).")
            cur.execute(_DROP_SQL)
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        for notice in raw.notices:
            logger.info("db: {}", notice.strip())
    finally:
        raw.autocommit = False
        fairy.close()
    logger.info("Schema applied from {}", SCHEMA_PATH)


def verify() -> None:
    with get_sync_engine().connect() as conn:
        tables = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name
        """)).scalars().all()
        timescale = conn.execute(text(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
        )).scalar()
        n_nodes = conn.execute(text("SELECT COUNT(*) FROM grid_nodes")).scalar()
    print(f"Tables     : {', '.join(tables)}")
    print(f"TimescaleDB: {timescale or 'not installed (plain Postgres mode)'}")
    print(f"Grid nodes : {n_nodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the AQ engine schema.")
    parser.add_argument("--reset", action="store_true", help="Drop project tables first.")
    parser.add_argument("--resolution", type=float, default=GRID_RESOLUTION_DEG)
    args = parser.parse_args()

    apply_schema(reset=args.reset)
    upsert_nodes(build_node_table(args.resolution))
    verify()
