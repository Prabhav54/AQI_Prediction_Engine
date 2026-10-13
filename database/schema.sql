-- ================================================================
-- database/schema.sql
-- Pan-India AQ Engine — portable Postgres / TimescaleDB schema
-- ================================================================
-- Idempotent: safe to re-run. Apply with
--   python database/setup_db.py            (uses DATABASE_URL_SYNC)
--
-- Runs on plain Postgres (e.g. Neon) and upgrades itself when the
-- TimescaleDB extension is available: hypertables, compression and a
-- continuous aggregate are created inside guarded DO blocks.
--
-- All CPCB math (24h rolling means, 8h rolling means, sub-indices,
-- composite AQI, category) runs in-database via refresh_aqi().
-- ================================================================

-- ── Table 0: grid_nodes (dimension) ─────────────────────────────
CREATE TABLE IF NOT EXISTS grid_nodes (
    location_hash   CHAR(8)          PRIMARY KEY,
    location_name   TEXT             NOT NULL,
    state           TEXT,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    kind            TEXT             NOT NULL DEFAULT 'grid',   -- 'city' | 'grid'
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_nodes_latlon ON grid_nodes (lat, lon);

-- ── Table 1: raw_observations (hourly weather + CAMS pollutants) ─
CREATE TABLE IF NOT EXISTS raw_observations (
    timestamp           TIMESTAMPTZ      NOT NULL,
    location_hash       CHAR(8)          NOT NULL,
    temp_c              REAL,
    humidity_pct        REAL,
    wind_speed_ms       REAL,
    wind_dir_deg        REAL,
    precip_mm           REAL,
    pressure_hpa        REAL,
    boundary_layer_m    REAL,
    radiation_wm2       REAL,
    pm25                REAL,            -- µg/m³
    pm10                REAL,            -- µg/m³
    no2                 REAL,            -- µg/m³
    so2                 REAL,            -- µg/m³
    co                  REAL,            -- mg/m³ (CPCB unit)
    o3                  REAL,            -- µg/m³
    dust                REAL,            -- µg/m³
    aod                 REAL,
    pm_gap_filled       BOOLEAN          NOT NULL DEFAULT FALSE,  -- PM from AOD proxy model
    ingested_at         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (timestamp, location_hash)
);

-- ── Table 2: aqi_computed (CPCB AQI per node-hour) ──────────────
CREATE TABLE IF NOT EXISTS aqi_computed (
    timestamp           TIMESTAMPTZ      NOT NULL,
    location_hash       CHAR(8)          NOT NULL,
    pm25_24h_avg        REAL,
    pm10_24h_avg        REAL,
    no2_24h_avg         REAL,
    so2_24h_avg         REAL,
    co_8h_avg           REAL,
    o3_8h_avg           REAL,
    sub_index_pm25      REAL,
    sub_index_pm10      REAL,
    sub_index_no2       REAL,
    sub_index_so2       REAL,
    sub_index_co        REAL,
    sub_index_o3        REAL,
    aqi                 SMALLINT,
    prominent_pollutant TEXT,
    aqi_category        TEXT,
    computed_at         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (timestamp, location_hash)
);

-- ── Table 3: aqi_forecasts (24h ensemble output per run) ────────
CREATE TABLE IF NOT EXISTS aqi_forecasts (
    forecast_generated_at   TIMESTAMPTZ  NOT NULL,   -- issue time (last observed hour)
    forecast_target_time    TIMESTAMPTZ  NOT NULL,
    location_hash           CHAR(8)      NOT NULL,
    hours_ahead             SMALLINT     NOT NULL,
    pm25                    REAL,        -- hourly forecast concentrations
    pm10                    REAL,
    no2                     REAL,
    so2                     REAL,
    co                      REAL,
    o3                      REAL,
    aqi                     SMALLINT     NOT NULL,   -- CPCB AQI on history + forecast
    aqi_lower               SMALLINT,                -- 80% interval from validation residuals
    aqi_upper               SMALLINT,
    aqi_category            TEXT,
    prominent_pollutant     TEXT,
    aqi_lstm                SMALLINT,                -- per-member AQI (explainability)
    aqi_gru                 SMALLINT,
    aqi_lgbm                SMALLINT,
    model_version           TEXT,
    PRIMARY KEY (forecast_generated_at, forecast_target_time, location_hash)
);

-- ── Table 4: forecast_skill (live backtest of every issued run) ─
CREATE TABLE IF NOT EXISTS forecast_skill (
    forecast_generated_at   TIMESTAMPTZ  NOT NULL,
    hours_ahead             SMALLINT     NOT NULL,
    n_nodes                 INTEGER      NOT NULL,
    aqi_mae                 REAL,
    aqi_rmse                REAL,
    aqi_bias                REAL,
    category_accuracy       REAL,        -- share of nodes where CPCB category was right
    persistence_aqi_mae     REAL,        -- baseline: "AQI stays what it was at issue time"
    pm25_mae                REAL,
    model_version           TEXT,
    evaluated_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (forecast_generated_at, hours_ahead)
);

-- ── Table 5: model_registry (deployed model cards) ───────────────
CREATE TABLE IF NOT EXISTS model_registry (
    model_version   TEXT         PRIMARY KEY,
    card            JSONB        NOT NULL,
    first_used_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_used_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ── Indices ──────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_raw_obs_location_time ON raw_observations (location_hash, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_aqi_location_time ON aqi_computed (location_hash, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_location ON aqi_forecasts (location_hash, forecast_generated_at DESC);

-- ── TimescaleDB upgrades (skipped silently on plain Postgres) ───
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
        CREATE EXTENSION IF NOT EXISTS timescaledb;
        PERFORM create_hypertable('raw_observations', 'timestamp',
            chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE, migrate_data => TRUE);
        PERFORM create_hypertable('aqi_computed', 'timestamp',
            chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE, migrate_data => TRUE);
        PERFORM create_hypertable('aqi_forecasts', 'forecast_generated_at',
            chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE, migrate_data => TRUE);
        RAISE NOTICE 'TimescaleDB hypertables enabled.';
    ELSE
        RAISE NOTICE 'TimescaleDB not available — using plain Postgres tables.';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'TimescaleDB setup skipped: %', SQLERRM;
END $$;

-- Compression is a Timescale-licensed feature (absent on Apache-only builds such as Neon)
DO $$
BEGIN
    ALTER TABLE raw_observations SET (timescaledb.compress, timescaledb.compress_segmentby = 'location_hash');
    PERFORM add_compression_policy('raw_observations', INTERVAL '7 days', if_not_exists => TRUE);
    ALTER TABLE aqi_computed SET (timescaledb.compress, timescaledb.compress_segmentby = 'location_hash');
    PERFORM add_compression_policy('aqi_computed', INTERVAL '7 days', if_not_exists => TRUE);
    RAISE NOTICE 'Compression policies enabled.';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Compression skipped: %', SQLERRM;
END $$;

-- ================================================================
-- CPCB math (single source of truth — mirrored by utils.cpcb_aqi_frame
-- for forecast hours, and checked against it in tests)
-- ================================================================

-- Piecewise-linear CPCB sub-index. `bp` holds the 7 concentration
-- breakpoints that map onto AQI 0, 50, 100, 200, 300, 400, 500.
CREATE OR REPLACE FUNCTION cpcb_sub_index(c DOUBLE PRECISION, bp DOUBLE PRECISION[])
RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN c IS NULL THEN NULL
        WHEN c >= bp[7] THEN 500.0
        ELSE (
            SELECT idx[i] + (GREATEST(c, 0) - bp[i]) * (idx[i + 1] - idx[i]) / (bp[i + 1] - bp[i])
            FROM (SELECT ARRAY[0, 50, 100, 200, 300, 400, 500]::DOUBLE PRECISION[] AS idx) s,
                 generate_series(1, 6) AS i
            WHERE GREATEST(c, 0) >= bp[i] AND GREATEST(c, 0) < bp[i + 1]
            LIMIT 1
        )
    END
$$;

CREATE OR REPLACE FUNCTION cpcb_category(aqi DOUBLE PRECISION)
RETURNS TEXT
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN aqi IS NULL THEN NULL
        WHEN aqi <= 50  THEN 'Good'
        WHEN aqi <= 100 THEN 'Satisfactory'
        WHEN aqi <= 200 THEN 'Moderate'
        WHEN aqi <= 300 THEN 'Poor'
        WHEN aqi <= 400 THEN 'Very Poor'
        ELSE 'Severe'
    END
$$;

-- Recompute AQI for every node-hour at or after `since`.
-- Window functions read 24h of extra history so the first rolling means are complete.
CREATE OR REPLACE FUNCTION refresh_aqi(since TIMESTAMPTZ)
RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE
    n INTEGER;
BEGIN
    WITH win AS (
        SELECT
            timestamp, location_hash,
            AVG(pm25) OVER w24 AS pm25_24h_avg,
            AVG(pm10) OVER w24 AS pm10_24h_avg,
            AVG(no2)  OVER w24 AS no2_24h_avg,
            AVG(so2)  OVER w24 AS so2_24h_avg,
            AVG(co)   OVER w8  AS co_8h_avg,
            AVG(o3)   OVER w8  AS o3_8h_avg
        FROM raw_observations
        WHERE timestamp >= since - INTERVAL '24 hours'
        WINDOW
            w24 AS (PARTITION BY location_hash ORDER BY timestamp
                    RANGE BETWEEN INTERVAL '23 hours' PRECEDING AND CURRENT ROW),
            w8  AS (PARTITION BY location_hash ORDER BY timestamp
                    RANGE BETWEEN INTERVAL '7 hours' PRECEDING AND CURRENT ROW)
    ),
    si AS (
        SELECT *,
            cpcb_sub_index(pm25_24h_avg, ARRAY[0, 30, 60, 90, 120, 250, 380]::DOUBLE PRECISION[])  AS s_pm25,
            cpcb_sub_index(pm10_24h_avg, ARRAY[0, 50, 100, 250, 350, 430, 600]::DOUBLE PRECISION[]) AS s_pm10,
            cpcb_sub_index(no2_24h_avg,  ARRAY[0, 40, 80, 180, 280, 400, 800]::DOUBLE PRECISION[])  AS s_no2,
            cpcb_sub_index(so2_24h_avg,  ARRAY[0, 40, 80, 380, 800, 1600, 2100]::DOUBLE PRECISION[]) AS s_so2,
            cpcb_sub_index(co_8h_avg,    ARRAY[0, 1, 2, 10, 17, 34, 50]::DOUBLE PRECISION[])        AS s_co,
            cpcb_sub_index(o3_8h_avg,    ARRAY[0, 50, 100, 168, 208, 748, 1000]::DOUBLE PRECISION[]) AS s_o3
        FROM win
        WHERE timestamp >= since
    ),
    final AS (
        SELECT *, GREATEST(s_pm25, s_pm10, s_no2, s_so2, s_co, s_o3) AS aqi_raw
        FROM si
    )
    INSERT INTO aqi_computed (
        timestamp, location_hash,
        pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg, co_8h_avg, o3_8h_avg,
        sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3,
        aqi, prominent_pollutant, aqi_category, computed_at
    )
    SELECT
        timestamp, location_hash,
        pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg, co_8h_avg, o3_8h_avg,
        s_pm25, s_pm10, s_no2, s_so2, s_co, s_o3,
        ROUND(aqi_raw)::SMALLINT,
        CASE aqi_raw
            WHEN s_pm25 THEN 'PM2.5' WHEN s_pm10 THEN 'PM10' WHEN s_no2 THEN 'NO2'
            WHEN s_so2  THEN 'SO2'   WHEN s_co   THEN 'CO'   WHEN s_o3  THEN 'O3'
        END,
        cpcb_category(ROUND(aqi_raw)),
        NOW()
    FROM final
    WHERE aqi_raw IS NOT NULL
    ON CONFLICT (timestamp, location_hash) DO UPDATE SET
        pm25_24h_avg = EXCLUDED.pm25_24h_avg, pm10_24h_avg = EXCLUDED.pm10_24h_avg,
        no2_24h_avg  = EXCLUDED.no2_24h_avg,  so2_24h_avg  = EXCLUDED.so2_24h_avg,
        co_8h_avg    = EXCLUDED.co_8h_avg,    o3_8h_avg    = EXCLUDED.o3_8h_avg,
        sub_index_pm25 = EXCLUDED.sub_index_pm25, sub_index_pm10 = EXCLUDED.sub_index_pm10,
        sub_index_no2  = EXCLUDED.sub_index_no2,  sub_index_so2  = EXCLUDED.sub_index_so2,
        sub_index_co   = EXCLUDED.sub_index_co,   sub_index_o3   = EXCLUDED.sub_index_o3,
        aqi = EXCLUDED.aqi, prominent_pollutant = EXCLUDED.prominent_pollutant,
        aqi_category = EXCLUDED.aqi_category, computed_at = NOW();
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n;
END
$$;

-- Score every forecast run whose 24h horizon has fully elapsed against the
-- AQI that actually materialised, plus a persistence baseline.
CREATE OR REPLACE FUNCTION evaluate_forecast_skill()
RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE
    n INTEGER;
BEGIN
    WITH pending AS (
        SELECT DISTINCT f.forecast_generated_at
        FROM aqi_forecasts f
        WHERE f.forecast_generated_at <= NOW() - INTERVAL '25 hours'
          AND NOT EXISTS (SELECT 1 FROM forecast_skill s
                          WHERE s.forecast_generated_at = f.forecast_generated_at)
    ),
    joined AS (
        SELECT f.forecast_generated_at, f.hours_ahead, f.model_version,
               f.aqi AS pred, a.aqi AS actual,
               f.aqi_category AS pred_cat, a.aqi_category AS actual_cat,
               f.pm25 AS pred_pm25, r.pm25 AS actual_pm25,
               a0.aqi AS issue_aqi
        FROM aqi_forecasts f
        JOIN pending p USING (forecast_generated_at)
        JOIN aqi_computed a
          ON a.location_hash = f.location_hash AND a.timestamp = f.forecast_target_time
        LEFT JOIN raw_observations r
          ON r.location_hash = f.location_hash AND r.timestamp = f.forecast_target_time
        LEFT JOIN aqi_computed a0
          ON a0.location_hash = f.location_hash AND a0.timestamp = f.forecast_generated_at
    )
    INSERT INTO forecast_skill (
        forecast_generated_at, hours_ahead, n_nodes, aqi_mae, aqi_rmse, aqi_bias,
        category_accuracy, persistence_aqi_mae, pm25_mae, model_version
    )
    SELECT forecast_generated_at, hours_ahead, COUNT(*),
           AVG(ABS(pred - actual)),
           SQRT(AVG((pred - actual)::DOUBLE PRECISION ^ 2)),
           AVG(pred - actual),
           AVG((pred_cat = actual_cat)::INT),
           AVG(ABS(issue_aqi - actual)),
           AVG(ABS(pred_pm25 - actual_pm25)),
           MAX(model_version)
    FROM joined
    GROUP BY forecast_generated_at, hours_ahead
    ON CONFLICT DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n;
END
$$;

-- ── Serving views ───────────────────────────────────────────────
-- Latest AQI per node (map layer)
CREATE OR REPLACE VIEW v_latest_aqi AS
SELECT DISTINCT ON (a.location_hash)
       a.location_hash, n.location_name, n.state, n.lat, n.lon, n.kind,
       a.timestamp, a.aqi, a.aqi_category, a.prominent_pollutant,
       a.pm25_24h_avg, a.pm10_24h_avg, a.no2_24h_avg, a.so2_24h_avg, a.co_8h_avg, a.o3_8h_avg,
       a.sub_index_pm25, a.sub_index_pm10, a.sub_index_no2,
       a.sub_index_so2, a.sub_index_co, a.sub_index_o3
FROM aqi_computed a
JOIN grid_nodes n USING (location_hash)
WHERE a.timestamp >= NOW() - INTERVAL '2 days'
ORDER BY a.location_hash, a.timestamp DESC;

-- Latest forecast run per node
CREATE OR REPLACE VIEW v_latest_forecast AS
SELECT f.*
FROM aqi_forecasts f
JOIN (
    SELECT location_hash, MAX(forecast_generated_at) AS forecast_generated_at
    FROM aqi_forecasts
    WHERE forecast_generated_at >= NOW() - INTERVAL '2 days'
    GROUP BY location_hash
) latest USING (location_hash, forecast_generated_at);

-- Daily AQI rollup: a Timescale continuous aggregate when available, else a plain view
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'aqi_daily') THEN
        EXECUTE $q$
            CREATE MATERIALIZED VIEW aqi_daily WITH (timescaledb.continuous) AS
            SELECT time_bucket('1 day', timestamp) AS day, location_hash,
                   AVG(aqi)::REAL AS aqi_mean, MAX(aqi) AS aqi_max, AVG(pm25_24h_avg)::REAL AS pm25_mean
            FROM aqi_computed
            GROUP BY 1, 2
            WITH NO DATA
        $q$;
        PERFORM add_continuous_aggregate_policy('aqi_daily',
            start_offset => INTERVAL '3 days', end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
        RAISE NOTICE 'Continuous aggregate aqi_daily created.';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Continuous aggregate unavailable (%), creating plain view.', SQLERRM;
    EXECUTE $q$
        CREATE OR REPLACE VIEW aqi_daily AS
        SELECT date_trunc('day', timestamp) AS day, location_hash,
               AVG(aqi)::REAL AS aqi_mean, MAX(aqi) AS aqi_max, AVG(pm25_24h_avg)::REAL AS pm25_mean
        FROM aqi_computed
        GROUP BY 1, 2
    $q$;
END $$;
