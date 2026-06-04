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

-- STEP 1: Calculate the 24-hour rolling averages and 8-hour maximums
CREATE OR REPLACE VIEW v_rolling_averages AS
SELECT 
    timestamp,
    location_hash,
    location_name,
    lat,
    lon,
    AVG(pm25_proxy) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) as pm25_24h_avg,
    AVG(pm10_proxy) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) as pm10_24h_avg,
    AVG(no2) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) as no2_24h_avg,
    AVG(so2) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) as so2_24h_avg,
    MAX(co) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '8 hours' PRECEDING AND CURRENT ROW) as co_8h_max,
    MAX(o3) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '8 hours' PRECEDING AND CURRENT ROW) as o3_8h_max
FROM raw_observations;

-- STEP 2: Calculate the Indian CPCB Sub-Indices for each pollutant
CREATE OR REPLACE VIEW v_sub_indices AS
SELECT 
    *,
    -- PM2.5 Sub-Index
    CASE 
        WHEN pm25_24h_avg <= 30 THEN (pm25_24h_avg * 50 / 30)
        WHEN pm25_24h_avg <= 60 THEN 50 + ((pm25_24h_avg - 30) * 50 / 30)
        WHEN pm25_24h_avg <= 90 THEN 100 + ((pm25_24h_avg - 60) * 100 / 30)
        WHEN pm25_24h_avg <= 120 THEN 200 + ((pm25_24h_avg - 90) * 100 / 30)
        WHEN pm25_24h_avg <= 250 THEN 300 + ((pm25_24h_avg - 120) * 100 / 130)
        ELSE 400 + ((pm25_24h_avg - 250) * 100 / 100)
    END as sub_index_pm25,
    
    -- PM10 Sub-Index
    CASE 
        WHEN pm10_24h_avg <= 50 THEN (pm10_24h_avg * 50 / 50)
        WHEN pm10_24h_avg <= 100 THEN 50 + ((pm10_24h_avg - 50) * 50 / 50)
        WHEN pm10_24h_avg <= 250 THEN 100 + ((pm10_24h_avg - 100) * 100 / 150)
        WHEN pm10_24h_avg <= 350 THEN 200 + ((pm10_24h_avg - 250) * 100 / 100)
        WHEN pm10_24h_avg <= 430 THEN 300 + ((pm10_24h_avg - 350) * 100 / 80)
        ELSE 400 + ((pm10_24h_avg - 430) * 100 / 100)
    END as sub_index_pm10,

    -- NO2 Sub-Index
    CASE 
        WHEN no2_24h_avg <= 40 THEN (no2_24h_avg * 50 / 40)
        WHEN no2_24h_avg <= 80 THEN 50 + ((no2_24h_avg - 40) * 50 / 40)
        WHEN no2_24h_avg <= 180 THEN 100 + ((no2_24h_avg - 80) * 100 / 100)
        WHEN no2_24h_avg <= 280 THEN 200 + ((no2_24h_avg - 180) * 100 / 100)
        WHEN no2_24h_avg <= 400 THEN 300 + ((no2_24h_avg - 280) * 100 / 120)
        ELSE 400 + ((no2_24h_avg - 400) * 100 / 100)
    END as sub_index_no2,

    -- SO2 Sub-Index
    CASE 
        WHEN so2_24h_avg <= 40 THEN (so2_24h_avg * 50 / 40)
        WHEN so2_24h_avg <= 80 THEN 50 + ((so2_24h_avg - 40) * 50 / 40)
        WHEN so2_24h_avg <= 380 THEN 100 + ((so2_24h_avg - 80) * 100 / 300)
        WHEN so2_24h_avg <= 800 THEN 200 + ((so2_24h_avg - 380) * 100 / 420)
        ELSE 300 + ((so2_24h_avg - 800) * 100 / 800)
    END as sub_index_so2,

    -- CO Sub-Index
    CASE 
        WHEN co_8h_max <= 1 THEN (co_8h_max * 50 / 1)
        WHEN co_8h_max <= 2 THEN 50 + ((co_8h_max - 1) * 50 / 1)
        WHEN co_8h_max <= 10 THEN 100 + ((co_8h_max - 2) * 100 / 8)
        WHEN co_8h_max <= 17 THEN 200 + ((co_8h_max - 10) * 100 / 7)
        WHEN co_8h_max <= 34 THEN 300 + ((co_8h_max - 17) * 100 / 17)
        ELSE 400 + ((co_8h_max - 34) * 100 / 100)
    END as sub_index_co,

    -- O3 Sub-Index
    CASE 
        WHEN o3_8h_max <= 50 THEN (o3_8h_max * 50 / 50)
        WHEN o3_8h_max <= 100 THEN 50 + ((o3_8h_max - 50) * 50 / 50)
        WHEN o3_8h_max <= 168 THEN 100 + ((o3_8h_max - 100) * 100 / 68)
        WHEN o3_8h_max <= 208 THEN 200 + ((o3_8h_max - 168) * 100 / 40)
        WHEN o3_8h_max <= 748 THEN 300 + ((o3_8h_max - 208) * 100 / 540)
        ELSE 400 + ((o3_8h_max - 748) * 100 / 100)
    END as sub_index_o3
FROM v_rolling_averages;

-- STEP 3: Combine sub-indices to extract the overall AQI, prominent pollutant, and category
CREATE OR REPLACE VIEW v_aqi_final AS
SELECT 
    *,
    -- Overall AQI is the maximum of individual sub-indices
    GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) as aqi,
    
    -- Determine prominent pollutant
    CASE GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3)
        WHEN sub_index_pm25 THEN 'PM2.5'
        WHEN sub_index_pm10 THEN 'PM10'
        WHEN sub_index_no2 THEN 'NO2'
        WHEN sub_index_so2 THEN 'SO2'
        WHEN sub_index_co THEN 'CO'
        WHEN sub_index_o3 THEN 'O3'
    END as prominent_pollutant,
    
    -- Assign regulatory category
    CASE 
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 50 THEN 'Good'
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 100 THEN 'Satisfactory'
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 200 THEN 'Moderate'
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 300 THEN 'Poor'
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 400 THEN 'Very Poor'
        ELSE 'Severe'
    END as aqi_category
FROM v_sub_indices;