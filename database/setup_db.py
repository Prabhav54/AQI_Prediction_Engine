"""
database/setup_db.py
--------------------
Cross-platform database setup script (works on Windows, Mac, Linux).
This replaces setup_db.sh which only runs on Unix/Mac.

What this does in order:
  1. Pulls and starts the TimescaleDB Docker container
  2. Creates the database user (aq_user)
  3. Creates the database (air_quality_db)
  4. Grants privileges
  5. Enables the TimescaleDB extension
  6. Applies database/schema.sql (creates all 3 hypertables)
  7. Verifies everything worked

Run:
  conda activate aq_engine
  python database/setup_db.py

Prerequisites:
  - Docker Desktop must be running (check the system tray icon)
  - Your .env file must exist with DB_* variables filled
"""

import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Load .env ───────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import os

DB_HOST      = os.getenv("DB_HOST",     "localhost")
DB_PORT      = os.getenv("DB_PORT",     "5432")
DB_NAME      = os.getenv("DB_NAME",     "air_quality_db")
DB_USER      = os.getenv("DB_USER",     "aq_user")
DB_PASS      = os.getenv("DB_PASS",     "aq_pass")
SUPER_USER   = "postgres"   # TimescaleDB Docker default superuser

SCHEMA_PATH  = Path(__file__).parent / "schema.sql"


# ================================================================
# Helper: run a shell command and print output
# ================================================================

def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess command with consistent error handling."""
    result = subprocess.run(
        cmd,
        capture_output = capture,
        text           = True,
    )
    if check and result.returncode != 0:
        err = result.stderr or result.stdout or "Unknown error"
        print(f"  ❌ Command failed: {' '.join(cmd)}")
        print(f"     {err.strip()}")
        sys.exit(1)
    return result


def docker_exec_psql(user: str, db: str, sql: str) -> subprocess.CompletedProcess:
    """Run a SQL command inside the timescaledb Docker container."""
    return run([
        "docker", "exec", "-i", "timescaledb",
        "psql", "-U", user, "-d", db, "-c", sql,
    ], check=False, capture=True)


def docker_exec_psql_file(user: str, db: str, filepath: str) -> subprocess.CompletedProcess:
    """Run a SQL file inside the timescaledb Docker container via stdin."""
    with open(filepath, "r") as f:
        sql_content = f.read()

    result = subprocess.run(
        ["docker", "exec", "-i", "timescaledb", "psql", "-U", user, "-d", db],
        input          = sql_content,
        capture_output = True,
        text           = True,
    )
    return result


# ================================================================
# Step 1 — Check Docker is running
# ================================================================

def check_docker() -> None:
    print("\n🐳 Checking Docker...")
    result = run(["docker", "info"], check=False, capture=True)
    if result.returncode != 0:
        print("  ❌ Docker is not running.")
        print("     Open Docker Desktop from the Start Menu / taskbar and wait for it to start.")
        sys.exit(1)
    print("  ✅ Docker is running.")


# ================================================================
# Step 2 — Start or create TimescaleDB container
# ================================================================

def start_timescaledb() -> None:
    print("\n⏱️  Setting up TimescaleDB container...")

    # Check if container already exists
    result = run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture=True, check=False
    )
    existing = result.stdout.strip().splitlines()

    if "timescaledb" in existing:
        print("  Container 'timescaledb' already exists — starting it...")
        run(["docker", "start", "timescaledb"], check=False)
    else:
        print("  Pulling timescale/timescaledb:latest-pg16 and starting...")
        run([
            "docker", "run", "-d",
            "--name", "timescaledb",
            "-p", f"{DB_PORT}:5432",
            "-e", f"POSTGRES_PASSWORD={DB_PASS}",
            "-v", "timescaledb_data:/var/lib/postgresql/data",
            "timescale/timescaledb:latest-pg16",
        ])

    # Wait for PostgreSQL to be ready
    print("  Waiting for PostgreSQL to accept connections", end="", flush=True)
    for attempt in range(20):
        time.sleep(2)
        check = docker_exec_psql(SUPER_USER, "postgres", "SELECT 1;")
        if check.returncode == 0:
            print(" ✅")
            return
        print(".", end="", flush=True)

    print("\n  ❌ PostgreSQL did not become ready in time. Try running the script again.")
    sys.exit(1)


# ================================================================
# Step 3 — Create user
# ================================================================

def create_user() -> None:
    print(f"\n👤 Creating user '{DB_USER}'...")

    # Check if user already exists
    check = docker_exec_psql(
        SUPER_USER, "postgres",
        f"SELECT 1 FROM pg_roles WHERE rolname='{DB_USER}';"
    )
    if "1 row" in (check.stdout or ""):
        print(f"  User '{DB_USER}' already exists — skipping.")
        return

    result = docker_exec_psql(
        SUPER_USER, "postgres",
        f"CREATE USER {DB_USER} WITH PASSWORD '{DB_PASS}';"
    )
    if result.returncode == 0:
        print(f"  ✅ User '{DB_USER}' created.")
    else:
        print(f"  ⚠️  User creation returned: {result.stderr.strip()}")


# ================================================================
# Step 4 — Create database
# ================================================================

def create_database() -> None:
    print(f"\n🗄️  Creating database '{DB_NAME}'...")

    check = docker_exec_psql(
        SUPER_USER, "postgres",
        f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}';"
    )
    if "1 row" in (check.stdout or ""):
        print(f"  Database '{DB_NAME}' already exists — skipping.")
        return

    result = docker_exec_psql(
        SUPER_USER, "postgres",
        f"CREATE DATABASE {DB_NAME} OWNER {DB_USER};"
    )
    if result.returncode == 0:
        print(f"  ✅ Database '{DB_NAME}' created.")
    else:
        print(f"  ⚠️  {result.stderr.strip()}")

    # Grant all privileges
    docker_exec_psql(
        SUPER_USER, DB_NAME,
        f"GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER}; "
        f"GRANT ALL ON SCHEMA public TO {DB_USER};"
    )
    print(f"  ✅ Privileges granted to '{DB_USER}'.")


# ================================================================
# Step 5 — Enable TimescaleDB extension
# ================================================================

def enable_timescaledb() -> None:
    print("\n⚙️  Enabling TimescaleDB extension...")

    result = docker_exec_psql(
        DB_USER, DB_NAME,
        "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
    )
    if result.returncode == 0:
        print("  ✅ TimescaleDB extension enabled.")
    else:
        # May need superuser — try with postgres
        result2 = docker_exec_psql(
            SUPER_USER, DB_NAME,
            "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
        )
        if result2.returncode == 0:
            print("  ✅ TimescaleDB extension enabled (via superuser).")
        else:
            print(f"  ❌ Failed: {result2.stderr.strip()}")
            sys.exit(1)


# ================================================================
# Step 6 — Apply schema
# ================================================================

def apply_schema() -> None:
    print(f"\n📐 Applying schema from {SCHEMA_PATH.name}...")

    if not SCHEMA_PATH.exists():
        print(f"  ❌ Schema file not found at: {SCHEMA_PATH}")
        print("     Make sure you're running this from the project root.")
        sys.exit(1)

    result = docker_exec_psql_file(DB_USER, DB_NAME, str(SCHEMA_PATH))

    if result.returncode == 0:
        print("  ✅ Schema applied successfully.")
    else:
        # TimescaleDB sometimes prints notices on schema apply — check for real errors
        stderr = result.stderr or ""
        if "ERROR" in stderr:
            print(f"  ❌ Schema error:\n{stderr}")
            sys.exit(1)
        else:
            print(f"  ✅ Schema applied (with notices).")


# ================================================================
# Step 7 — Verify: list hypertables
# ================================================================

def verify() -> None:
    print("\n🔍 Verifying hypertables...")

    result = docker_exec_psql(
        DB_USER, DB_NAME,
        "SELECT hypertable_name FROM timescaledb_information.hypertables;"
    )

    expected = {"raw_observations", "aqi_computed", "aqi_forecasts"}

    if result.returncode != 0:
        print(f"  ❌ Verification failed: {result.stderr.strip()}")
        return

    output = result.stdout or ""
    found  = {name.strip() for name in output.splitlines() if name.strip() in expected}
    missing = expected - found

    if not missing:
        print(f"  ✅ All 3 hypertables found: {', '.join(sorted(found))}")
    else:
        print(f"  ⚠️  Missing hypertables: {missing}")
        print("     Try running: python database/setup_db.py  again.")


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Pan-India AQ Engine — Database Setup")
    print(f"  Host    : {DB_HOST}:{DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User    : {DB_USER}")
    print("=" * 55)

    check_docker()
    start_timescaledb()
    create_user()
    create_database()
    enable_timescaledb()
    apply_schema()
    verify()

    print("\n" + "=" * 55)
    print("✅  Database setup complete!")
    print()
    print("  Connection string (async / FastAPI):")
    print(f"  postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    print()
    print("  Next step — verify the full setup:")
    print("  python verify_setup.py --skip-gee")
    print("=" * 55 + "\n")