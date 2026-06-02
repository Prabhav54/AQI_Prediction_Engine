import pandas as pd
import random
from datetime import datetime, timedelta, timezone

def load_ensemble_weights():
    """
    Returns the designated blending weights.
    Requested: 0.6 LSTM + 0.4 XGBoost
    """
    return {"lstm": 0.6, "xgb": 0.4}

def _get_aqi_category(aqi: int) -> str:
    """Helper function to map Indian AQI numbers to CPCB categories."""
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

def ensemble_forecast_24h(sequence_df: pd.DataFrame, forecast_hours: int = 24):
    """
    Simulates the ensemble forecast logic using the 0.6/0.4 split.
    (Replace the random generation with real model inference later).
    """
    weights = load_ensemble_weights()
    alpha = weights["lstm"]  # 0.6
    
    forecasts = []
    
    # Start time at the top of the current hour
    base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    
    # Starting baseline AQI for the simulation
    current_trend = 145 

    for i in range(1, forecast_hours + 1):
        target_time = base_time + timedelta(hours=i)
        
        # 1. Simulate the two models predicting slightly different futures
        lstm_pred = int(current_trend + random.uniform(-8, 8))
        xgb_pred = int(current_trend + random.uniform(-12, 12))
        
        # 2. Apply your requested ensemble math: (0.6 * LSTM) + (0.4 * XGBoost)
        blended_aqi = int((lstm_pred * alpha) + (xgb_pred * (1 - alpha)))
        
        # 3. Package it into the exact dictionary format expected by forecast.py
        forecasts.append({
            "forecast_target_time": target_time,
            "hours_ahead": i,
            "aqi_forecast": blended_aqi,
            "aqi_category_forecast": _get_aqi_category(blended_aqi),
            "aqi_lstm": lstm_pred,
            "aqi_xgb": xgb_pred,
            "ensemble_alpha": alpha
        })
        
        # Shift the trend slightly for the next hour to create a realistic curve
        current_trend = blended_aqi

    return forecasts