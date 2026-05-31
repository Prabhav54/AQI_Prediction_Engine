-- ================================================================
-- database/schema.sql
-- TimescaleDB Schema — Pan-India AQ Engine
-- ================================================================
-- Run this ONCE after creating the database:
--   psql -h localhost -U aq_user -d air_quality_db -f database/schema.sql
--
-- Prerequisites:
--   1. PostgreSQL 14+ with TimescaleDB extension installed
--   2. Database and user already created (see setup_db.sh)
-- ================================================================


-- ── Enable TimescaleDB extension ────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;


-- ================================================================
-- TABLE 1: raw_observations
-- Stores the merged output of Module 1 (ingestion pipeline).
-- One row per (location, hour). This is the raw landing table.
-- ================================================================
CREATE TABLE IF NOT EXISTS raw_observations (
    -- Time dimension (TimescaleDB partitions on this column)
    timestamp           TIMESTAMPTZ         NOT NULL,

    -- Location
    lat                 DOUBLE PRECISION    NOT NULL,
    lon                 DOUBLE PRECISION    NOT NULL,
    location_name       TEXT                NOT NULL,
    location_hash       CHAR(8)             NOT NULL,   -- MD5 of rounded lat/lon

    -- Satellite columns (Sentinel-5P, µmol/m² or mmol/m²)
    -- NULLs are expected — satellite has cloud gaps
    no2                 DOUBLE PRECISION,   -- tropospheric NO₂ column (µmol/m²)
    so2                 DOUBLE PRECISION,   -- SO₂ column (µmol/m²)
    co                  DOUBLE PRECISION,   -- CO column (mmol/m²)
    o3                  DOUBLE PRECISION,   -- O₃ column (mmol/m²)
    aod                 DOUBLE PRECISION,   -- MODIS AOD @ 047nm (dimensionless)

    -- Weather columns (Open-Meteo)
    -- These should never be NULL after ingestion
    temp_c              DOUBLE PRECISION    NOT NULL,
    humidity_pct        DOUBLE PRECISION    NOT NULL,
    wind_speed_ms       DOUBLE PRECISION    NOT NULL,
    precip_mm           DOUBLE PRECISION    DEFAULT 0,
    pressure_hpa        DOUBLE PRECISION,
    boundary_layer_m    DOUBLE PRECISION,

    -- Proxy model outputs (Module 2) — populated after ETL
    pm25_proxy          DOUBLE PRECISION,   -- estimated PM2.5 (µg/m³)
    pm10_proxy          DOUBLE PRECISION,   -- estimated PM10 (µg/m³)

    -- Audit
    ingested_at         TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    pipeline_version    TEXT                DEFAULT '0.1.0',

    -- Composite primary key: one reading per location-hour
    PRIMARY KEY (timestamp, location_hash)
);

-- Convert to TimescaleDB hypertable, partitioned by time
-- chunk_time_interval = 1 day is appropriate for hourly AQ data
SELECT create_hypertable(
    'raw_observations',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);


-- ================================================================
-- TABLE 2: aqi_computed
-- Stores the CPCB AQI results computed by the SQL in aqi_sql.sql.
-- Populated by the continuous aggregation job.
-- ================================================================
CREATE TABLE IF NOT EXISTS aqi_computed (
    timestamp               TIMESTAMPTZ     NOT NULL,
    location_hash           CHAR(8)         NOT NULL,
    location_name           TEXT            NOT NULL,
    lat                     DOUBLE PRECISION NOT NULL,
    lon                     DOUBLE PRECISION NOT NULL,

    -- 24-hour rolling averages (inputs to CPCB sub-index formulas)
    pm25_24h_avg            DOUBLE PRECISION,
    pm10_24h_avg            DOUBLE PRECISION,
    no2_24h_avg             DOUBLE PRECISION,
    so2_24h_avg             DOUBLE PRECISION,

    -- 8-hour maximums
    co_8h_max               DOUBLE PRECISION,
    o3_8h_max               DOUBLE PRECISION,

    -- Individual CPCB sub-indices (I_p for each pollutant)
    sub_index_pm25          DOUBLE PRECISION,
    sub_index_pm10          DOUBLE PRECISION,
    sub_index_no2           DOUBLE PRECISION,
    sub_index_so2           DOUBLE PRECISION,
    sub_index_co            DOUBLE PRECISION,
    sub_index_o3            DOUBLE PRECISION,

    -- Final AQI = GREATEST of all sub-indices
    aqi                     INTEGER,
    prominent_pollutant     TEXT,           -- e.g. 'PM25', 'NO2'
    aqi_category            TEXT,           -- Good / Satisfactory / Moderate / Poor / Very Poor / Severe

    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (timestamp, location_hash)
);

SELECT create_hypertable(
    'aqi_computed',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);


-- ================================================================
-- TABLE 3: aqi_forecasts
-- Stores LSTM 24-hour ahead forecasts (Module 4 output).
-- ================================================================
CREATE TABLE IF NOT EXISTS aqi_forecasts (
    -- The time at which the forecast was GENERATED
    forecast_generated_at   TIMESTAMPTZ     NOT NULL,

    -- The time being PREDICTED (T+1 to T+24)
    forecast_target_time    TIMESTAMPTZ     NOT NULL,

    location_hash           CHAR(8)         NOT NULL,
    location_name           TEXT            NOT NULL,
    lat                     DOUBLE PRECISION NOT NULL,
    lon                     DOUBLE PRECISION NOT NULL,

    -- Forecast values
    aqi_forecast            DOUBLE PRECISION NOT NULL,
    aqi_category_forecast   TEXT,

    -- Confidence interval (populated if model outputs uncertainty)
    aqi_lower_95            DOUBLE PRECISION,
    aqi_upper_95            DOUBLE PRECISION,

    -- Model metadata
    model_version           TEXT            DEFAULT '0.1.0',
    lookback_hours_used     INTEGER         DEFAULT 168,

    PRIMARY KEY (forecast_generated_at, forecast_target_time, location_hash)
);

SELECT create_hypertable(
    'aqi_forecasts',
    'forecast_generated_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);


-- ================================================================
-- INDEXES
-- ================================================================

-- Most queries filter by location_hash + time range
CREATE INDEX IF NOT EXISTS idx_raw_obs_location_time
    ON raw_observations (location_hash, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_aqi_location_time
    ON aqi_computed (location_hash, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_forecasts_location_generated
    ON aqi_forecasts (location_hash, forecast_generated_at DESC);

-- For the Streamlit map view — spatial bounding-box queries
CREATE INDEX IF NOT EXISTS idx_raw_obs_latlon
    ON raw_observations (lat, lon);


-- ================================================================
-- COMPRESSION POLICY
-- TimescaleDB compresses chunks older than 7 days.
-- Reduces storage by ~90% for time-series numerical data.
-- ================================================================
ALTER TABLE raw_observations SET (
    timescaledb.compress,
    timescaledb.compress_orderby    = 'timestamp DESC',
    timescaledb.compress_segmentby  = 'location_hash'
);

SELECT add_compression_policy(
    'raw_observations',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE aqi_computed SET (
    timescaledb.compress,
    timescaledb.compress_orderby    = 'timestamp DESC',
    timescaledb.compress_segmentby  = 'location_hash'
);

SELECT add_compression_policy(
    'aqi_computed',
    INTERVAL '7 days',
    if_not_exists => TRUE
);


-- ================================================================
-- RETENTION POLICY
-- Auto-drop raw observations older than 90 days.
-- Computed AQI and forecasts kept for 1 year.
-- ================================================================
SELECT add_retention_policy(
    'raw_observations',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'aqi_forecasts',
    INTERVAL '365 days',
    if_not_exists => TRUE
);


-- ================================================================
-- VALIDATION — run these after schema creation to confirm setup
-- ================================================================
-- SELECT * FROM timescaledb_information.hypertables;
-- SELECT * FROM timescaledb_information.jobs;
-- \dt  -- should show 3 tables