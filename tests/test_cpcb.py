"""
tests/test_cpcb.py
------------------
CPCB AQI math. The SQL implementation (refresh_aqi in schema.sql) is canonical;
these tests pin the Python mirror to known values and — when a database is
reachable — to the SQL output on the same synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from utils import aqi_along_paths, aqi_category, compute_aqi, cpcb_aqi_frame, cpcb_sub_index


@pytest.mark.parametrize("conc,expected", [
    (0, 0), (30, 50), (45, 75), (60, 100), (90, 200), (250, 400), (380, 500), (900, 500),
])
def test_pm25_breakpoints(conc, expected):
    assert float(cpcb_sub_index(conc, "pm25")) == pytest.approx(expected)


def test_nan_stays_nan_and_negative_clips_to_zero():
    out = cpcb_sub_index([np.nan, -5.0], "no2")
    assert np.isnan(out[0]) and out[1] == 0


def test_compute_aqi_picks_prominent():
    r = compute_aqi({"pm25": 45, "pm10": 80, "no2": 30, "so2": 10, "co": 0.8, "o3": 40})
    assert r["aqi"] == 80 and r["prominent_pollutant"] == "PM10" and r["category"] == "Satisfactory"


@pytest.mark.parametrize("aqi,cat", [(50, "Good"), (51, "Satisfactory"), (200, "Moderate"),
                                     (301, "Very Poor"), (401, "Severe")])
def test_categories(aqi, cat):
    assert aqi_category(aqi) == cat


def _synthetic_series(hours=72, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "pm25": rng.gamma(4, 20, hours), "pm10": rng.gamma(4, 35, hours),
        "no2": rng.gamma(3, 12, hours), "so2": rng.gamma(2, 8, hours),
        "co": rng.gamma(2, 0.6, hours), "o3": rng.gamma(3, 25, hours),
    }, index=idx)


def test_paths_match_frame():
    """aqi_along_paths (forecast scoring) == cpcb_aqi_frame (reference) on the same series."""
    s = _synthetic_series()
    ref = cpcb_aqi_frame(s)
    cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    t = 40
    hist = s[cols].values[t - 22:t + 1][None]          # 23 hours up to t
    fut = s[cols].values[t + 1:t + 25][None]           # t+1..t+24
    aqi, _ = aqi_along_paths(hist, fut)
    np.testing.assert_allclose(np.round(aqi[0]), ref["aqi"].values[t + 1:t + 25])


def test_sql_matches_python():
    """refresh_aqi() in Postgres must agree with the Python mirror."""
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from database.db_client import get_sync_engine

    try:
        engine = get_sync_engine()
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT refresh_aqi(NOW())"))
    except Exception:
        pytest.skip("No database with the project schema available.")

    s = _synthetic_series(hours=60, seed=1)
    h = "test0001"
    rows = s.reset_index().rename(columns={"index": "timestamp"})
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("DELETE FROM raw_observations WHERE location_hash = :h"), {"h": h})
        conn.execute(sqlalchemy.text("DELETE FROM aqi_computed WHERE location_hash = :h"), {"h": h})
        for r in rows.itertuples():
            conn.execute(sqlalchemy.text("""
                INSERT INTO raw_observations (timestamp, location_hash, pm25, pm10, no2, so2, co, o3)
                VALUES (:t, :h, :pm25, :pm10, :no2, :so2, :co, :o3)
            """), dict(t=r.timestamp, h=h, pm25=r.pm25, pm10=r.pm10, no2=r.no2,
                       so2=r.so2, co=r.co, o3=r.o3))
        conn.execute(sqlalchemy.text("SELECT refresh_aqi(:t)"), {"t": rows["timestamp"].min()})
        sql = pd.read_sql(sqlalchemy.text(
            "SELECT timestamp, aqi FROM aqi_computed WHERE location_hash = :h ORDER BY timestamp"
        ), conn, params={"h": h})
        conn.execute(sqlalchemy.text("DELETE FROM raw_observations WHERE location_hash = :h"), {"h": h})
        conn.execute(sqlalchemy.text("DELETE FROM aqi_computed WHERE location_hash = :h"), {"h": h})

    py = cpcb_aqi_frame(s.astype("float32").astype(float))["aqi"].values
    # REAL storage → allow ±1 rounding difference
    assert np.abs(sql["aqi"].values - py).max() <= 1
