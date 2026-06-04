"""
ui/app.py
---------
Sleek, Production-Ready Streamlit Frontend.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_BASE = "http://localhost:8000"

# Modern Neon Colors for Dark Mode
CATEGORY_COLOURS = {
    "Good": "#00E676", "Satisfactory": "#FFEA00", "Moderate": "#FF9100",
    "Poor": "#FF1744", "Very Poor": "#D500F9", "Severe": "#880E4F",
}

st.set_page_config(page_title="AQI Prediction Engine", page_icon="🌤️", layout="wide")

def call_api(method: str, path: str, **kwargs):
    try:
        resp = getattr(requests, method)(f"{API_BASE}{path}", timeout=15, **kwargs)
        if resp.status_code == 404: return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def main():
    # Minimalist Header
    st.markdown("<h1 style='text-align: center;'>🌍 Air Quality Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Real-time PM2.5 tracking with 24-hour LSTM + XGBoost ensemble forecasts</p>", unsafe_allow_html=True)
    st.write("")

    # Clean Search Bar centered on the screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        location = st.text_input("Search any city:", placeholder="e.g. Goa, Tokyo, Mumbai", label_visibility="collapsed")
        fetch_btn = st.button("🚀 Analyze Air Quality", use_container_width=True)

    if fetch_btn and location.strip():
        with st.spinner(f"Scanning atmosphere above {location.strip()}..."):
            # Because the API bypasses the satellite now, this will take < 3 seconds!
            result = call_api("post", "/ingest/", json={"location": location.strip(), "lookback_days": 7})
            
            if result:
                st.session_state["lat"] = result["lat"]
                st.session_state["lon"] = result["lon"]
                st.session_state["loc_name"] = result.get("location", location.title())
                
                # Tiny buffer to let the database save the rows before we fetch the graphs
                time.sleep(2) 
                st.rerun()
            else:
                st.error("❌ Failed to fetch data. Ensure your backend API is running.")

    st.divider()

    # --- RESULTS DASHBOARD ---
    lat = st.session_state.get("lat")
    if not lat:
        st.markdown("<h4 style='text-align: center; color: gray; margin-top: 50px;'>Enter a city above to view real-time pollution metrics.</h4>", unsafe_allow_html=True)
        return

    loc_name = st.session_state.get("loc_name", "Unknown")
    st.markdown(f"<h2 style='text-align: center;'>📍 {loc_name}</h2>", unsafe_allow_html=True)
    
    # Fetch Data for the graphs
    aqi_data = call_api("get", f"/aqi?lat={lat}&lon={st.session_state['lon']}")
    forecast_data = call_api("get", f"/forecast?lat={lat}&lon={st.session_state['lon']}")

    if not aqi_data or not forecast_data:
        st.warning("Crunching numbers... Please wait a second and click Analyze again.")
        return

    # Visual Layout
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Current AQI")
        aqi = aqi_data["aqi"]
        cat = aqi_data["aqi_category"]
        color = CATEGORY_COLOURS.get(cat, "gray")
        
        # Custom Sleek Metric Box
        st.markdown(
            f"""
            <div style="background-color: #1E1E1E; padding: 30px; border-radius: 15px; text-align: center; border-bottom: 5px solid {color};">
                <h1 style="font-size: 70px; margin: 0; color: white;">{aqi}</h1>
                <h3 style="color: {color}; margin: 5px 0;">{cat.upper()}</h3>
                <p style="color: gray; font-size: 14px;">Primary Pollutant: <b style="color: white;">{aqi_data['prominent_pollutant']}</b></p>
            </div>
            """, unsafe_allow_html=True
        )

        st.write("")
        st.subheader("Pollutant Levels")
        p = aqi_data.get("pollutants", {})
        df_p = pd.DataFrame({
            "Pollutant": ["PM2.5", "PM10", "NO₂", "O₃", "SO₂"],
            "Conc.": [
                f"{p.get('pm25_24h_avg', 0):.1f} µg/m³",
                f"{p.get('pm10_24h_avg', 0):.1f} µg/m³",
                f"{p.get('no2_24h_avg', 0):.1f} µg/m³",
                f"{p.get('o3_8h_max', 0):.1f} µg/m³",
                f"{p.get('so2_24h_avg', 0):.1f} µg/m³",
            ]
        })
        st.dataframe(df_p, hide_index=True, use_container_width=True)

    with c2:
        st.subheader("24-Hour AI Forecast")
        hourly = forecast_data["hourly"]
        
        # Format the time beautifully
        times = [h["forecast_target_time"][11:16] for h in hourly] 
        ensembles = [h["aqi_forecast"] for h in hourly]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=ensembles, mode='lines+markers',
            line=dict(color='#00d4ff', width=3),
            marker=dict(size=8, color='white', line=dict(color='#00d4ff', width=2)),
            fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)',
            name="Predicted AQI"
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#333'),
            height=380,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        alpha = forecast_data.get("ensemble_alpha", 0.6)
        st.caption(f"🧠 Powered by Ensemble AI (LSTM: {alpha:.0%} | XGBoost: {1-alpha:.0%})")

if __name__ == "__main__":
    main()