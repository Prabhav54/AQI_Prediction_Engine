-- ================================================================
-- database/aqi_sql.sql
-- Module 3 — CPCB AQI Computation via SQL Window Functions
-- ================================================================
-- This is where all the heavy AQI math happens.
-- We do it in SQL rather than Python because:
--   1. TimescaleDB can apply this incrementally as new data arrives
--   2. Window functions over millions of rows are faster in the DB
--   3. The AQI formula is deterministic — no reason to move data
--      out of the DB just to compute it and push it back
--
-- What this script does, step by step:
--   Step 1 → 24-hour rolling averages for PM2.5, PM10, NO2, SO2
--   Step 2 → 8-hour rolling maximums for CO and O3
--   Step 3 → CPCB linear interpolation for each pollutant's sub-index
--   Step 4 → GREATEST() across all sub-indices = final AQI
--   Step 5 → Write results into the aqi_computed hypertable
--
-- Run manually:
--   psql -U aq_user -d air_quality_db -f database/aqi_sql.sql
--
-- In production this runs as a TimescaleDB continuous aggregate
-- job that refreshes every hour automatically.
-- ================================================================


-- ================================================================
-- STEP 1 — Rolling averages and maximums using window functions
-- ================================================================
-- We create a view so the aggregation logic is reusable.
-- The db_client.py queries this view directly.

CREATE OR REPLACE VIEW v_rolling_aggregates AS
SELECT
    timestamp,
    location_hash,
    location_name,
    lat,
    lon,

    -- 24-hour rolling averages (CPCB standard for PM2.5, PM10, NO2, SO2)
    AVG(pm25_proxy)  OVER w24 AS pm25_24h_avg,
    AVG(pm10_proxy)  OVER w24 AS pm10_24h_avg,
    AVG(no2)         OVER w24 AS no2_24h_avg,
    AVG(so2)         OVER w24 AS so2_24h_avg,

    -- 8-hour rolling maximums (CPCB standard for CO and O3)
    -- We use MAX here because peak exposure matters more for these gases
    MAX(co)          OVER w8  AS co_8h_max,
    MAX(o3)          OVER w8  AS o3_8h_max,

    -- Keep the latest raw values too — useful for the frontend
    pm25_proxy       AS pm25_current,
    pm10_proxy       AS pm10_current,
    temp_c,
    humidity_pct,
    wind_speed_ms

FROM raw_observations

-- 24-hour window: current row + 23 preceding hours, ordered by time
WINDOW
    w24 AS (
        PARTITION BY location_hash
        ORDER BY timestamp
        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    ),
    -- 8-hour window: current row + 7 preceding hours
    w8 AS (
        PARTITION BY location_hash
        ORDER BY timestamp
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    );


-- ================================================================
-- STEP 2 — CPCB Sub-Index Calculation
-- ================================================================
-- Official formula (CPCB, November 2014):
--
--   I_p = [(I_hi - I_lo) / (Bp_hi - Bp_lo)] × (C_p - Bp_lo) + I_lo
--
-- Where:
--   C_p   = measured concentration
--   Bp_lo = lower breakpoint concentration for that AQI sub-range
--   Bp_hi = upper breakpoint concentration
--   I_lo  = lower AQI value for that sub-range
--   I_hi  = upper AQI value for that sub-range
--
-- Each pollutant has 6 breakpoint ranges mapping to AQI 0-500.
-- The CASE WHEN chain below implements this for each pollutant.
-- ================================================================

CREATE OR REPLACE VIEW v_sub_indices AS
SELECT
    timestamp,
    location_hash,
    location_name,
    lat,
    lon,
    pm25_24h_avg,
    pm10_24h_avg,
    no2_24h_avg,
    so2_24h_avg,
    co_8h_max,
    o3_8h_max,

    -- ── PM2.5 Sub-Index (24h average, µg/m³) ──────────────────
    CASE
        WHEN pm25_24h_avg IS NULL                       THEN NULL
        WHEN pm25_24h_avg BETWEEN 0    AND 30   THEN ROUND(((50  - 0  ) / (30   - 0  )) * (pm25_24h_avg - 0  ) + 0  )
        WHEN pm25_24h_avg BETWEEN 31   AND 60   THEN ROUND(((100 - 51 ) / (60   - 31 )) * (pm25_24h_avg - 31 ) + 51 )
        WHEN pm25_24h_avg BETWEEN 61   AND 90   THEN ROUND(((200 - 101) / (90   - 61 )) * (pm25_24h_avg - 61 ) + 101)
        WHEN pm25_24h_avg BETWEEN 91   AND 120  THEN ROUND(((300 - 201) / (120  - 91 )) * (pm25_24h_avg - 91 ) + 201)
        WHEN pm25_24h_avg BETWEEN 121  AND 250  THEN ROUND(((400 - 301) / (250  - 121)) * (pm25_24h_avg - 121) + 301)
        WHEN pm25_24h_avg > 250                         THEN LEAST(ROUND(((500 - 401) / (380 - 251)) * (pm25_24h_avg - 251) + 401), 500)
    END AS sub_index_pm25,

    -- ── PM10 Sub-Index (24h average, µg/m³) ───────────────────
    CASE
        WHEN pm10_24h_avg IS NULL                       THEN NULL
        WHEN pm10_24h_avg BETWEEN 0    AND 50   THEN ROUND(((50  - 0  ) / (50   - 0  )) * (pm10_24h_avg - 0  ) + 0  )
        WHEN pm10_24h_avg BETWEEN 51   AND 100  THEN ROUND(((100 - 51 ) / (100  - 51 )) * (pm10_24h_avg - 51 ) + 51 )
        WHEN pm10_24h_avg BETWEEN 101  AND 250  THEN ROUND(((200 - 101) / (250  - 101)) * (pm10_24h_avg - 101) + 101)
        WHEN pm10_24h_avg BETWEEN 251  AND 350  THEN ROUND(((300 - 201) / (350  - 251)) * (pm10_24h_avg - 251) + 201)
        WHEN pm10_24h_avg BETWEEN 351  AND 430  THEN ROUND(((400 - 301) / (430  - 351)) * (pm10_24h_avg - 351) + 301)
        WHEN pm10_24h_avg > 430                         THEN LEAST(ROUND(((500 - 401) / (600 - 431)) * (pm10_24h_avg - 431) + 401), 500)
    END AS sub_index_pm10,

    -- ── NO2 Sub-Index (24h average, µg/m³) ────────────────────
    CASE
        WHEN no2_24h_avg IS NULL                        THEN NULL
        WHEN no2_24h_avg BETWEEN 0    AND 40    THEN ROUND(((50  - 0  ) / (40   - 0  )) * (no2_24h_avg - 0  ) + 0  )
        WHEN no2_24h_avg BETWEEN 41   AND 80    THEN ROUND(((100 - 51 ) / (80   - 41 )) * (no2_24h_avg - 41 ) + 51 )
        WHEN no2_24h_avg BETWEEN 81   AND 180   THEN ROUND(((200 - 101) / (180  - 81 )) * (no2_24h_avg - 81 ) + 101)
        WHEN no2_24h_avg BETWEEN 181  AND 280   THEN ROUND(((300 - 201) / (280  - 181)) * (no2_24h_avg - 181) + 201)
        WHEN no2_24h_avg BETWEEN 281  AND 400   THEN ROUND(((400 - 301) / (400  - 281)) * (no2_24h_avg - 281) + 301)
        WHEN no2_24h_avg > 400                          THEN LEAST(ROUND(((500 - 401) / (800 - 401)) * (no2_24h_avg - 401) + 401), 500)
    END AS sub_index_no2,

    -- ── SO2 Sub-Index (24h average, µg/m³) ────────────────────
    CASE
        WHEN so2_24h_avg IS NULL                        THEN NULL
        WHEN so2_24h_avg BETWEEN 0    AND 40    THEN ROUND(((50  - 0  ) / (40   - 0  )) * (so2_24h_avg - 0  ) + 0  )
        WHEN so2_24h_avg BETWEEN 41   AND 80    THEN ROUND(((100 - 51 ) / (80   - 41 )) * (so2_24h_avg - 41 ) + 51 )
        WHEN so2_24h_avg BETWEEN 81   AND 380   THEN ROUND(((200 - 101) / (380  - 81 )) * (so2_24h_avg - 81 ) + 101)
        WHEN so2_24h_avg BETWEEN 381  AND 800   THEN ROUND(((300 - 201) / (800  - 381)) * (so2_24h_avg - 381) + 201)
        WHEN so2_24h_avg BETWEEN 801  AND 1600  THEN ROUND(((400 - 301) / (1600 - 801)) * (so2_24h_avg - 801) + 301)
        WHEN so2_24h_avg > 1600                         THEN LEAST(ROUND(((500 - 401) / (2100 - 1601)) * (so2_24h_avg - 1601) + 401), 500)
    END AS sub_index_so2,

    -- ── CO Sub-Index (8h maximum, mg/m³) ──────────────────────
    CASE
        WHEN co_8h_max IS NULL                          THEN NULL
        WHEN co_8h_max BETWEEN 0    AND 1.0     THEN ROUND(((50  - 0  ) / (1.0  - 0   )) * (co_8h_max - 0   ) + 0  )
        WHEN co_8h_max BETWEEN 1.1  AND 2.0     THEN ROUND(((100 - 51 ) / (2.0  - 1.1 )) * (co_8h_max - 1.1 ) + 51 )
        WHEN co_8h_max BETWEEN 2.1  AND 10.0    THEN ROUND(((200 - 101) / (10.0 - 2.1 )) * (co_8h_max - 2.1 ) + 101)
        WHEN co_8h_max BETWEEN 10.1 AND 17.0    THEN ROUND(((300 - 201) / (17.0 - 10.1)) * (co_8h_max - 10.1) + 201)
        WHEN co_8h_max BETWEEN 17.1 AND 34.0    THEN ROUND(((400 - 301) / (34.0 - 17.1)) * (co_8h_max - 17.1) + 301)
        WHEN co_8h_max > 34.0                           THEN LEAST(ROUND(((500 - 401) / (50.0 - 34.1)) * (co_8h_max - 34.1) + 401), 500)
    END AS sub_index_co,

    -- ── O3 Sub-Index (8h maximum, µg/m³) ──────────────────────
    CASE
        WHEN o3_8h_max IS NULL                          THEN NULL
        WHEN o3_8h_max BETWEEN 0    AND 50      THEN ROUND(((50  - 0  ) / (50   - 0  )) * (o3_8h_max - 0  ) + 0  )
        WHEN o3_8h_max BETWEEN 51   AND 100     THEN ROUND(((100 - 51 ) / (100  - 51 )) * (o3_8h_max - 51 ) + 51 )
        WHEN o3_8h_max BETWEEN 101  AND 168     THEN ROUND(((200 - 101) / (168  - 101)) * (o3_8h_max - 101) + 101)
        WHEN o3_8h_max BETWEEN 169  AND 208     THEN ROUND(((300 - 201) / (208  - 169)) * (o3_8h_max - 169) + 201)
        WHEN o3_8h_max BETWEEN 209  AND 748     THEN ROUND(((400 - 301) / (748  - 209)) * (o3_8h_max - 209) + 301)
        WHEN o3_8h_max > 748                            THEN LEAST(ROUND(((500 - 401) / (1000 - 749)) * (o3_8h_max - 749) + 401), 500)
    END AS sub_index_o3

FROM v_rolling_aggregates;


-- ================================================================
-- STEP 3 — Final AQI: GREATEST of all sub-indices
-- ================================================================
-- Per CPCB definition:
--   AQI = the highest individual sub-index among all pollutants
--   Prominent Pollutant = the pollutant with that highest sub-index
--
-- We also attach a human-readable category label here so the
-- API and frontend don't need to compute it themselves.
-- ================================================================

CREATE OR REPLACE VIEW v_aqi_final AS
SELECT
    timestamp,
    location_hash,
    location_name,
    lat,
    lon,

    -- Raw pollutant averages (for the frontend detail view)
    pm25_24h_avg,
    pm10_24h_avg,
    no2_24h_avg,
    so2_24h_avg,
    co_8h_max,
    o3_8h_max,

    -- Individual sub-indices
    sub_index_pm25,
    sub_index_pm10,
    sub_index_no2,
    sub_index_so2,
    sub_index_co,
    sub_index_o3,

    -- Final AQI = max of all available sub-indices
    GREATEST(
        COALESCE(sub_index_pm25, 0),
        COALESCE(sub_index_pm10, 0),
        COALESCE(sub_index_no2,  0),
        COALESCE(sub_index_so2,  0),
        COALESCE(sub_index_co,   0),
        COALESCE(sub_index_o3,   0)
    ) AS aqi,

    -- Prominent pollutant — whichever drove the AQI value
    -- Uses a CASE to find the pollutant matching the max sub-index
    CASE GREATEST(
            COALESCE(sub_index_pm25, 0),
            COALESCE(sub_index_pm10, 0),
            COALESCE(sub_index_no2,  0),
            COALESCE(sub_index_so2,  0),
            COALESCE(sub_index_co,   0),
            COALESCE(sub_index_o3,   0)
         )
        WHEN COALESCE(sub_index_pm25, 0) THEN 'PM2.5'
        WHEN COALESCE(sub_index_pm10, 0) THEN 'PM10'
        WHEN COALESCE(sub_index_no2,  0) THEN 'NO2'
        WHEN COALESCE(sub_index_so2,  0) THEN 'SO2'
        WHEN COALESCE(sub_index_co,   0) THEN 'CO'
        WHEN COALESCE(sub_index_o3,   0) THEN 'O3'
        ELSE 'Unknown'
    END AS prominent_pollutant,

    -- Human-readable AQI category (CPCB standard labels)
    CASE
        WHEN GREATEST(
            COALESCE(sub_index_pm25, 0), COALESCE(sub_index_pm10, 0),
            COALESCE(sub_index_no2,  0), COALESCE(sub_index_so2,  0),
            COALESCE(sub_index_co,   0), COALESCE(sub_index_o3,   0)
        ) <= 50   THEN 'Good'
        WHEN GREATEST(
            COALESCE(sub_index_pm25, 0), COALESCE(sub_index_pm10, 0),
            COALESCE(sub_index_no2,  0), COALESCE(sub_index_so2,  0),
            COALESCE(sub_index_co,   0), COALESCE(sub_index_o3,   0)
        ) <= 100  THEN 'Satisfactory'
        WHEN GREATEST(
            COALESCE(sub_index_pm25, 0), COALESCE(sub_index_pm10, 0),
            COALESCE(sub_index_no2,  0), COALESCE(sub_index_so2,  0),
            COALESCE(sub_index_co,   0), COALESCE(sub_index_o3,   0)
        ) <= 200  THEN 'Moderate'
        WHEN GREATEST(
            COALESCE(sub_index_pm25, 0), COALESCE(sub_index_pm10, 0),
            COALESCE(sub_index_no2,  0), COALESCE(sub_index_so2,  0),
            COALESCE(sub_index_co,   0), COALESCE(sub_index_o3,   0)
        ) <= 300  THEN 'Poor'
        WHEN GREATEST(
            COALESCE(sub_index_pm25, 0), COALESCE(sub_index_pm10, 0),
            COALESCE(sub_index_no2,  0), COALESCE(sub_index_so2,  0),
            COALESCE(sub_index_co,   0), COALESCE(sub_index_o3,   0)
        ) <= 400  THEN 'Very Poor'
        ELSE 'Severe'
    END AS aqi_category

FROM v_sub_indices;


-- ================================================================
-- STEP 4 — Materialise results into aqi_computed hypertable
-- ================================================================
-- This INSERT is called by db_client.py after every ingestion run.
-- ON CONFLICT updates existing rows so re-runs are idempotent.

INSERT INTO aqi_computed (
    timestamp, location_hash, location_name, lat, lon,
    pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg,
    co_8h_max, o3_8h_max,
    sub_index_pm25, sub_index_pm10, sub_index_no2,
    sub_index_so2, sub_index_co, sub_index_o3,
    aqi, prominent_pollutant, aqi_category,
    computed_at
)
SELECT
    timestamp, location_hash, location_name, lat, lon,
    pm25_24h_avg, pm10_24h_avg, no2_24h_avg, so2_24h_avg,
    co_8h_max, o3_8h_max,
    sub_index_pm25, sub_index_pm10, sub_index_no2,
    sub_index_so2, sub_index_co, sub_index_o3,
    aqi, prominent_pollutant, aqi_category,
    NOW()
FROM v_aqi_final
WHERE aqi > 0   -- skip rows where all sub-indices were NULL/zero
ON CONFLICT (timestamp, location_hash)
DO UPDATE SET
    aqi                = EXCLUDED.aqi,
    prominent_pollutant = EXCLUDED.prominent_pollutant,
    aqi_category       = EXCLUDED.aqi_category,
    pm25_24h_avg       = EXCLUDED.pm25_24h_avg,
    pm10_24h_avg       = EXCLUDED.pm10_24h_avg,
    computed_at        = NOW();