"""
forecasting/dataset.py
----------------------
Upgraded Spatially-Aware Dataset Loader for Pan-India Grid.
Extracts timeseries vectors from PostGIS and injects normalized coordinates.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from database.db_client import get_sync_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Global feature checklist matching ensemble model expectations
LSTM_FEATURE_COLS = [
    'aqi', 'pm25_24h_avg', 'pm10_24h_avg', 'no2_24h_avg', 'so2_24h_avg', 
    'co_8h_max', 'o3_8h_max', 'temp_c', 'humidity_pct', 'wind_speed_ms',
    'precip_mm', 'pressure_hpa', 'boundary_layer_m', 
    'hour_of_day', 'day_of_week', 'is_weekend', 'temp_change_6h', 'pm25_change_3h',
    'lat_norm', 'lon_norm'
]

N_FEATURES = len(LSTM_FEATURE_COLS)

def fetch_global_training_data(lookback_days: int = 30) -> pd.DataFrame:
    """
    Pulls historical tracking vectors from all active grid nodes simultaneously.
    Joins rolling target metrics with weather parameters.
    """
    engine = get_sync_engine()
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    
    query = text("""
        SELECT 
            a.timestamp,
            a.location_hash,
            a.lat,
            a.lon,
            a.aqi, 
            a.pm25_24h_avg, 
            a.pm10_24h_avg,
            a.no2_24h_avg, 
            a.so2_24h_avg, 
            a.co_8h_max, 
            a.o3_8h_max,
            r.temp_c, 
            r.humidity_pct, 
            r.wind_speed_ms,
            r.precip_mm, 
            r.pressure_hpa, 
            r.boundary_layer_m
        FROM aqi_computed a
        LEFT JOIN raw_observations r
            ON  a.timestamp     = r.timestamp
            AND a.location_hash = r.location_hash
        WHERE a.timestamp >= :since
        ORDER BY a.location_hash, a.timestamp ASC
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"since": since})
        logger.info(f"Successfully extracted {len(df)} row sequences for global optimization training.")
        return df
    except Exception as exc:
        logger.error(f"Failed to extract global training data matrix: {exc}")
        return pd.DataFrame()

def engineer_spatial_tensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cyclic temporal extensions and normalizes geographic spatial vectors
    so a single global model can differentiate between microclimates in India.
    """
    if df.empty:
        return df
        
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # 1. Core Time Feature Engineering
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend']  = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # 2. Dynamic Trend Momentum Indicators
    df['temp_change_6h'] = df.groupby('location_hash')['temp_c'].diff(6).fillna(0)
    df['pm25_change_3h'] = df.groupby('location_hash')['pm25_24h_avg'].diff(3).fillna(0)
    
    # 3. CRUCIAL: Spatial Coordinate Bounding Normalization (India Landmass)
    # Scales absolute degrees uniformly into relative [0, 1] relative feature coordinates
    df['lat_norm'] = (df['lat'] - 6.5) / (37.5 - 6.5)
    df['lon_norm'] = (df['lon'] - 68.5) / (97.5 - 68.5)
    
    return df

def prepare_lstm_sequences(df: pd.DataFrame, sequence_length: int = 168) -> tuple[np.ndarray, np.ndarray]:
    """
    Chunks the unified regional dataframe into sequential sliding windows for LSTM input matrices.
    """
    df_engineered = engineer_spatial_tensors(df)
    if df_engineered.empty:
        return np.array([]), np.array([])
        
    X, y = [], []
    
    # Build history sequence windows independently per location node hash code block
    for _, group in df_engineered.groupby('location_hash'):
        if len(group) < sequence_length + 24: # Require enough data points for 24h targets
            continue
            
        data = group[LSTM_FEATURE_COLS].values
        target = group['aqi'].values
        
        for i in range(len(data) - sequence_length - 24 + 1):
            X.append(data[i : i + sequence_length])
            y.append(target[i + sequence_length : i + sequence_length + 24])
            
    return np.array(X), np.array(y)