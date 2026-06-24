-- ================================================================
-- database/schema.sql
-- TimescaleDB + PostGIS Schema — Pan-India AQ Engine
-- ================================================================
-- Run this to clear and initialize your schema:
--   psql -h localhost -U aq_user -d air_quality_db -f database/schema.sql

-- ── Core Extensions ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis CASCADE;

-- ── Table 1: raw_observations (Hypertables + Spatial) ───────────
CREATE TABLE IF NOT EXISTS raw_observations (
    timestamp           TIMESTAMPTZ         NOT NULL,
    lat                 DOUBLE PRECISION    NOT NULL,
    lon                 DOUBLE PRECISION    NOT NULL,
    location_name       TEXT                NOT NULL,
    location_hash       CHAR(8)             NOT NULL,
    no2                 DOUBLE PRECISION,   
    so2                 DOUBLE PRECISION,   
    co                  DOUBLE PRECISION,   
    o3                  DOUBLE PRECISION,   
    aod                 DOUBLE PRECISION,   
    temp_c              DOUBLE PRECISION    NOT NULL,
    humidity_pct        DOUBLE PRECISION    NOT NULL,
    wind_speed_ms       DOUBLE PRECISION    NOT NULL,
    precip_mm           DOUBLE PRECISION    DEFAULT 0,
    pressure_hpa        DOUBLE PRECISION,
    boundary_layer_m    DOUBLE PRECISION,
    pm25_proxy          DOUBLE PRECISION,   
    pm10_proxy          DOUBLE PRECISION,   
    ingested_at         TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    pipeline_version    TEXT                DEFAULT '1.0.0',
    geom                GEOMETRY(Point, 4326),
    PRIMARY KEY (timestamp, location_hash)
);

SELECT create_hypertable(
    'raw_observations',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- ── Table 2: aqi_computed ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS aqi_computed (
    timestamp               TIMESTAMPTZ     NOT NULL,
    location_hash           CHAR(8)         NOT NULL,
    location_name           TEXT            NOT NULL,
    lat                     DOUBLE PRECISION NOT NULL,
    lon                     DOUBLE PRECISION NOT NULL,
    pm25_24h_avg            DOUBLE PRECISION,
    pm10_24h_avg            DOUBLE PRECISION,
    no2_24h_avg             DOUBLE PRECISION,
    so2_24h_avg             DOUBLE PRECISION,
    co_8h_max               DOUBLE PRECISION,
    o3_8h_max               DOUBLE PRECISION,
    sub_index_pm25          DOUBLE PRECISION,
    sub_index_pm10          DOUBLE PRECISION,
    sub_index_no2           DOUBLE PRECISION,
    sub_index_so2           DOUBLE PRECISION,
    sub_index_co            DOUBLE PRECISION,
    sub_index_o3            DOUBLE PRECISION,
    aqi                     INTEGER,
    prominent_pollutant     TEXT,           
    aqi_category            TEXT,           
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (timestamp, location_hash)
);

SELECT create_hypertable(
    'aqi_computed',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- ── Table 3: aqi_forecasts ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS aqi_forecasts (
    forecast_generated_at   TIMESTAMPTZ     NOT NULL,
    forecast_target_time    TIMESTAMPTZ     NOT NULL,
    location_hash           CHAR(8)         NOT NULL,
    location_name           TEXT            NOT NULL,
    lat                     DOUBLE PRECISION NOT NULL,
    lon                     DOUBLE PRECISION NOT NULL,
    aqi_forecast            DOUBLE PRECISION NOT NULL,
    aqi_category_forecast   TEXT,
    aqi_lower_95            DOUBLE PRECISION,
    aqi_upper_95            DOUBLE PRECISION,
    model_version           TEXT            DEFAULT '1.0.0',
    lookback_hours_used     INTEGER         DEFAULT 168,
    PRIMARY KEY (forecast_generated_at, forecast_target_time, location_hash)
);

SELECT create_hypertable(
    'aqi_forecasts',
    'forecast_generated_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

-- ── Indices ──────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_raw_obs_geom ON raw_observations USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_raw_obs_location_time ON raw_observations (location_hash, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_aqi_location_time ON aqi_computed (location_hash, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_location_generated ON aqi_forecasts (location_hash, forecast_generated_at DESC);

-- ── Compression Policies ──────────────────────────────────────────
ALTER TABLE raw_observations SET (timescaledb.compress, timescaledb.compress_segmentby = 'location_hash');
SELECT add_compression_policy('raw_observations', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE aqi_computed SET (timescaledb.compress, timescaledb.compress_segmentby = 'location_hash');
SELECT add_compression_policy('aqi_computed', INTERVAL '7 days', if_not_exists => TRUE);

-- ── Retention Policies ───────────────────────────────────────────
SELECT add_retention_policy('raw_observations', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('aqi_forecasts', INTERVAL '365 days', if_not_exists => TRUE);

-- ── Views for CPCB Calculations ──────────────────────────────────
CREATE OR REPLACE VIEW v_rolling_averages AS
SELECT 
    timestamp, location_hash, location_name, lat, lon,
    AVG(pm25_proxy) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) as pm25_24h_avg,
    AVG(pm10_proxy) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) as pm10_24h_avg,
    AVG(no2) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) as no2_24h_avg,
    AVG(so2) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) as so2_24h_avg,
    MAX(co) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '8 hours' PRECEDING AND CURRENT ROW) as co_8h_max,
    MAX(o3) OVER(PARTITION BY location_hash ORDER BY timestamp RANGE BETWEEN INTERVAL '8 hours' PRECEDING AND CURRENT ROW) as o3_8h_max
FROM raw_observations;

CREATE OR REPLACE VIEW v_sub_indices AS
SELECT 
    *,
    CASE 
        WHEN pm25_24h_avg <= 30 THEN (pm25_24h_avg * 50 / 30)
        WHEN pm25_24h_avg <= 60 THEN 50 + ((pm25_24h_avg - 30) * 50 / 30)
        WHEN pm25_24h_avg <= 90 THEN 100 + ((pm25_24h_avg - 60) * 100 / 30)
        WHEN pm25_24h_avg <= 120 THEN 200 + ((pm25_24h_avg - 90) * 100 / 30)
        WHEN pm25_24h_avg <= 250 THEN 300 + ((pm25_24h_avg - 120) * 100 / 130)
        ELSE 400 + ((pm25_24h_avg - 250) * 100 / 100)
    END as sub_index_pm25,
    CASE 
        WHEN pm10_24h_avg <= 50 THEN (pm10_24h_avg * 50 / 50)
        WHEN pm10_24h_avg <= 100 THEN 50 + ((pm10_24h_avg - 50) * 50 / 50)
        WHEN pm10_24h_avg <= 250 THEN 100 + ((pm10_24h_avg - 100) * 100 / 150)
        WHEN pm10_24h_avg <= 350 THEN 200 + ((pm10_24h_avg - 250) * 100 / 100)
        WHEN pm10_24h_avg <= 430 THEN 300 + ((pm10_24h_avg - 350) * 100 / 80)
        ELSE 400 + ((pm10_24h_avg - 430) * 100 / 100)
    END as sub_index_pm10,
    CASE 
        WHEN no2_24h_avg <= 40 THEN (no2_24h_avg * 50 / 40)
        WHEN no2_24h_avg <= 80 THEN 50 + ((no2_24h_avg - 40) * 50 / 40)
        WHEN no2_24h_avg <= 180 THEN 100 + ((no2_24h_avg - 80) * 100 / 100)
        WHEN no2_24h_avg <= 280 THEN 200 + ((no2_24h_avg - 180) * 100 / 100)
        WHEN no2_24h_avg <= 400 THEN 300 + ((no2_24h_avg - 280) * 100 / 120)
        ELSE 400 + ((no2_24h_avg - 400) * 100 / 100)
    END as sub_index_no2,
    CASE 
        WHEN so2_24h_avg <= 40 THEN (so2_24h_avg * 50 / 40)
        WHEN so2_24h_avg <= 80 THEN 50 + ((so2_24h_avg - 40) * 50 / 40)
        WHEN so2_24h_avg <= 380 THEN 100 + ((so2_24h_avg - 80) * 100 / 300)
        WHEN so2_24h_avg <= 800 THEN 200 + ((so2_24h_avg - 380) * 100 / 420)
        ELSE 300 + ((so2_24h_avg - 800) * 100 / 800)
    END as sub_index_so2,
    CASE 
        WHEN co_8h_max <= 1 THEN (co_8h_max * 50 / 1)
        WHEN co_8h_max <= 2 THEN 50 + ((co_8h_max - 1) * 50 / 1)
        WHEN co_8h_max <= 10 THEN 100 + ((co_8h_max - 2) * 100 / 8)
        WHEN co_8h_max <= 17 THEN 200 + ((co_8h_max - 10) * 100 / 7)
        WHEN co_8h_max <= 34 THEN 300 + ((co_8h_max - 17) * 100 / 17)
        ELSE 400 + ((co_8h_max - 34) * 100 / 100)
    END as sub_index_co,
    CASE 
        WHEN o3_8h_max <= 50 THEN (o3_8h_max * 50 / 50)
        WHEN o3_8h_max <= 100 THEN 50 + ((o3_8h_max - 50) * 50 / 50)
        WHEN o3_8h_max <= 168 THEN 100 + ((o3_8h_max - 100) * 100 / 68)
        WHEN o3_8h_max <= 208 THEN 200 + ((o3_8h_max - 168) * 100 / 40)
        WHEN o3_8h_max <= 748 THEN 300 + ((o3_8h_max - 208) * 100 / 540)
        ELSE 400 + ((o3_8h_max - 748) * 100 / 100)
    END as sub_index_o3
FROM v_rolling_averages;

CREATE OR REPLACE VIEW v_aqi_final AS
SELECT 
    *,
    GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) as aqi,
    CASE GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3)
        WHEN sub_index_pm25 THEN 'PM2.5'
        WHEN sub_index_pm10 THEN 'PM10'
        WHEN sub_index_no2 THEN 'NO2'
        WHEN sub_index_so2 THEN 'SO2'
        WHEN sub_index_co THEN 'CO'
        WHEN sub_index_o3 THEN 'O3'
    END as prominent_pollutant,
    CASE 
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 50 THEN 'Good'
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 100 THEN 'Satisfactory'
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 200 THEN 'Moderate'
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 300 THEN 'Poor'
        WHEN GREATEST(sub_index_pm25, sub_index_pm10, sub_index_no2, sub_index_so2, sub_index_co, sub_index_o3) <= 400 THEN 'Very Poor'
        ELSE 'Severe'
    END as aqi_category
FROM v_sub_indices;