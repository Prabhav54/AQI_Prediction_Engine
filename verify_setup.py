"""
verify_setup.py
---------------
Run this after completing all setup steps to confirm every
dependency (DB, GEE, network) is correctly configured.

Usage:
    conda activate aq_engine
    python verify_setup.py

    # Skip GEE check (no credentials yet):
    python verify_setup.py --skip-gee

    # Also write a summary report:
    python verify_setup.py --report
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

# ── Rich for pretty console output (installed via environment.yml) ──
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs): print(*args)
    console = Console()


PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️  SKIP"
WARN = "⚠️  WARN"

results: list[dict] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append({"name": name, "status": status, "detail": detail})
    colour = {"✅": "green", "❌": "red", "⚠️": "yellow", "⏭️": "dim"}
    icon = status[:2]
    clr = colour.get(icon, "white")
    if HAS_RICH:
        console.print(f"  [{clr}]{status}[/{clr}]  {name}", end="")
        if detail:
            console.print(f"  [dim]— {detail}[/dim]", end="")
        console.print()
    else:
        print(f"  {status}  {name}" + (f"  — {detail}" if detail else ""))


# ================================================================
# 1. Python version
# ================================================================
def check_python():
    console.print("\n[bold]1. Python Version[/bold]" if HAS_RICH else "\n1. Python Version")
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major == 3 and minor == 11:
        check("Python 3.11", PASS, version_str)
    elif major == 3 and minor in (10, 12):
        check("Python version", WARN,
              f"{version_str} — 3.11 recommended; some deps may behave differently")
    else:
        check("Python version", FAIL,
              f"{version_str} — this project requires Python 3.11")


# ================================================================
# 2. .env file
# ================================================================
def check_env():
    console.print("\n[bold]2. Environment File (.env)[/bold]" if HAS_RICH else "\n2. Environment File (.env)")
    if not os.path.exists(".env"):
        check(".env file", FAIL, "Not found. Run: cp .env.example .env")
        return

    from dotenv import load_dotenv
    load_dotenv(override=True)
    check(".env file", PASS, "Found and loaded")

    required_vars = [
        ("DATABASE_URL",       True,  "PostgreSQL connection string"),
        ("DATABASE_URL_SYNC",  True,  "Sync PostgreSQL URL for migrations"),
        ("GEE_SERVICE_ACCOUNT",False, "GEE service account email"),
        ("GEE_KEY_FILE",       False, "Path to GEE JSON key"),
        ("API_SECRET_KEY",     False, "FastAPI JWT secret"),
    ]

    for var, required, desc in required_vars:
        val = os.getenv(var, "")
        if val and "<REPLACE" not in val:
            check(f"  {var}", PASS, desc)
        elif required:
            check(f"  {var}", FAIL, f"{desc} — REQUIRED but not set")
        else:
            check(f"  {var}", WARN, f"{desc} — not set (optional for now)")


# ================================================================
# 3. Core Python packages
# ================================================================
def check_packages():
    console.print("\n[bold]3. Core Package Imports[/bold]" if HAS_RICH else "\n3. Core Package Imports")

    packages = [
        ("numpy",           "numpy",            True),
        ("pandas",          "pandas",            True),
        ("requests",        "requests",          True),
        ("python-dotenv",   "dotenv",            True),
        ("sqlalchemy",      "sqlalchemy",        True),
        ("asyncpg",         "asyncpg",           True),
        ("scikit-learn",    "sklearn",           True),
        ("xgboost",         "xgboost",           True),
        ("joblib",          "joblib",            True),
        ("fastapi",         "fastapi",           True),
        ("uvicorn",         "uvicorn",           True),
        ("streamlit",       "streamlit",         True),
        ("plotly",          "plotly",            True),
        ("loguru",          "loguru",            True),
        ("tenacity",        "tenacity",          True),
        ("torch",           "torch",             False),  # large install
        ("earthengine-api", "ee",                False),  # needs credentials
    ]

    for pkg_name, import_name, required in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            check(f"  {pkg_name}", PASS, f"v{version}")
        except ImportError:
            status = FAIL if required else WARN
            hint = "pip install " + pkg_name
            check(f"  {pkg_name}", status, f"Not installed — {hint}")


# ================================================================
# 4. Database connection
# ================================================================
def check_database():
    console.print("\n[bold]4. PostgreSQL + TimescaleDB[/bold]" if HAS_RICH else "\n4. PostgreSQL + TimescaleDB")

    db_url = os.getenv("DATABASE_URL_SYNC",
                       "postgresql://aq_user:aq_pass@localhost:5432/air_quality_db")

    try:
        import psycopg2
        conn = psycopg2.connect(db_url, connect_timeout=5)
        cursor = conn.cursor()

        # PostgreSQL version
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0].split(",")[0]
        check("PostgreSQL connection", PASS, pg_version)

        # TimescaleDB extension
        cursor.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"
        )
        row = cursor.fetchone()
        if row:
            check("TimescaleDB extension", PASS, f"v{row[0]}")
        else:
            check("TimescaleDB extension", FAIL,
                  "Not enabled. Run: CREATE EXTENSION IF NOT EXISTS timescaledb;")

        # Check hypertables exist
        cursor.execute(
            "SELECT hypertable_name FROM timescaledb_information.hypertables;"
        )
        hypertables = [r[0] for r in cursor.fetchall()]
        expected = {"raw_observations", "aqi_computed", "aqi_forecasts"}
        found = set(hypertables)

        if expected.issubset(found):
            check("Hypertables", PASS, ", ".join(sorted(found)))
        else:
            missing = expected - found
            check("Hypertables", WARN if not missing else FAIL,
                  f"Missing: {missing} — run: psql ... -f database/schema.sql")

        cursor.close()
        conn.close()

    except ImportError:
        check("psycopg2", FAIL, "pip install psycopg2-binary")
    except Exception as exc:
        check("PostgreSQL connection", FAIL,
              f"{exc} — is TimescaleDB running? Try: ./database/setup_db.sh")


# ================================================================
# 5. Open-Meteo (no auth — just a live network test)
# ================================================================
def check_open_meteo():
    console.print("\n[bold]5. Open-Meteo API (weather)[/bold]" if HAS_RICH else "\n5. Open-Meteo API (weather)")

    try:
        import requests
        # Tiny request: 1 hour of temperature for Delhi
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 28.6139,
                "longitude": 77.2090,
                "hourly": "temperature_2m",
                "forecast_days": 1,
                "timezone": "UTC",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        temps = data.get("hourly", {}).get("temperature_2m", [])
        check("Open-Meteo API", PASS,
              f"Reachable — sample Delhi temp: {temps[0]}°C" if temps else "Reachable")
    except Exception as exc:
        check("Open-Meteo API", FAIL, str(exc))


# ================================================================
# 6. Nominatim geocoding (no auth — just a live network test)
# ================================================================
def check_nominatim():
    console.print("\n[bold]6. Nominatim Geocoding[/bold]" if HAS_RICH else "\n6. Nominatim Geocoding")

    try:
        import requests
        time.sleep(1.1)   # respect rate limit
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": "Mumbai, India", "format": "json", "limit": 1},
            headers={"User-Agent": "pan_india_aq_engine/1.0 (verify_setup)"},
            timeout=10,
        )
        resp.raise_for_status()
        results_data = resp.json()
        if results_data:
            r = results_data[0]
            check("Nominatim geocoder", PASS,
                  f"Mumbai → ({float(r['lat']):.3f}, {float(r['lon']):.3f})")
        else:
            check("Nominatim geocoder", WARN, "No results returned for test query")
    except Exception as exc:
        check("Nominatim geocoder", FAIL, str(exc))


# ================================================================
# 7. Google Earth Engine
# ================================================================
def check_gee(skip: bool = False):
    console.print("\n[bold]7. Google Earth Engine[/bold]" if HAS_RICH else "\n7. Google Earth Engine")

    if skip:
        check("GEE auth", SKIP, "Skipped with --skip-gee")
        return

    sa  = os.getenv("GEE_SERVICE_ACCOUNT", "")
    key = os.getenv("GEE_KEY_FILE", "config/gee_key.json")

    if not sa or "<REPLACE" in sa:
        check("GEE_SERVICE_ACCOUNT", WARN,
              "Not set — set it in .env or use --skip-gee for mock mode")
        return

    if not os.path.exists(key):
        check(f"GEE key file ({key})", FAIL,
              "JSON key not found — download from GCP Console → Service Accounts → Keys")
        return

    try:
        import ee
        credentials = ee.ServiceAccountCredentials(sa, key)
        ee.Initialize(credentials)

        # Quick live test — get metadata of a known image
        img = ee.Image("NASA/NASADEM_HGT/001")
        info = img.getInfo()
        check("GEE authentication", PASS,
              f"Service account: {sa.split('@')[0]}@...")
        check("GEE live query", PASS, f"Test image: {info.get('id', 'OK')}")
    except ImportError:
        check("earthengine-api", FAIL, "pip install earthengine-api")
    except Exception as exc:
        check("GEE authentication", FAIL, str(exc))


# ================================================================
# 8. Print summary
# ================================================================
def print_summary(write_report: bool = False):
    console.print("\n" + "="*55 if not HAS_RICH else "")
    if HAS_RICH:
        console.print("\n[bold]── Setup Summary ─────────────────────────────────[/bold]")

    total  = len(results)
    passed = sum(1 for r in results if "PASS" in r["status"])
    failed = sum(1 for r in results if "FAIL" in r["status"])
    warned = sum(1 for r in results if "WARN" in r["status"])

    if HAS_RICH:
        table = Table(box=box.SIMPLE)
        table.add_column("Check",  style="cyan",   no_wrap=True)
        table.add_column("Status", style="white",  no_wrap=True)
        table.add_column("Detail", style="dim")
        for r in results:
            colour = "green" if "PASS" in r["status"] else \
                     "red"   if "FAIL" in r["status"] else \
                     "yellow"if "WARN" in r["status"] else "dim"
            table.add_row(r["name"], f"[{colour}]{r['status']}[/{colour}]", r["detail"])
        console.print(table)

    console.print(
        f"\n  Total: {total}  |  "
        f"Passed: {passed}  |  "
        f"Warnings: {warned}  |  "
        f"Failed: {failed}"
    )

    if failed == 0 and warned == 0:
        console.print("\n🚀 All checks passed! You're ready to run the pipeline.\n")
    elif failed == 0:
        console.print("\n⚠️  Setup mostly complete — review warnings above.\n")
    else:
        console.print("\n❌  Fix the FAILED checks before running the pipeline.\n")

    if write_report:
        report_path = f"setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, "w") as f:
            f.write(f"Pan-India AQ Engine — Setup Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            for r in results:
                f.write(f"{r['status']:12}  {r['name']:40}  {r['detail']}\n")
        console.print(f"📄 Report written to: {report_path}\n")


# ================================================================
# Entry point
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify Pan-India AQ Engine setup."
    )
    parser.add_argument(
        "--skip-gee", action="store_true",
        help="Skip Google Earth Engine credential check."
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Write a text summary report to disk."
    )
    args = parser.parse_args()

    if HAS_RICH:
        console.print(
            "\n[bold cyan]Pan-India AQ Engine — Setup Verification[/bold cyan]"
        )
    else:
        print("\nPan-India AQ Engine — Setup Verification")

    check_python()
    check_env()
    check_packages()
    check_database()
    check_open_meteo()
    check_nominatim()
    check_gee(skip=args.skip_gee)
    print_summary(write_report=args.report)