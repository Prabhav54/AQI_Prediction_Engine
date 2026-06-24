"""
ui/app.py
---------
Premium, High-Performance Streamlit Frontend for the Pan-India AQI Prediction Engine.
Features an ultra-clean single-page interface with hardcoded, production-optimized background values.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import geocoder and safety modules from your core engine layers
from ingestion.geocoder import geocode
from exceptions import LocationOutsideIndiaError, GeocodingError

API_BASE = "http://localhost:8000"

# Premium Minimalist Palette for Dark Mode
CATEGORY_COLOURS = {
    "Good": "#00E676",          # Vibrant Neon Green
    "Satisfactory": "#00B0FF",  # Clean Cyan Blue
    "Moderate": "#FFEA00",      # Warning Yellow
    "Poor": "#FF9100",          # Severe Amber Orange
    "Very Poor": "#FF1744",     # Critical Coral Red
    "Severe": "#D500F9",        # Extreme Purple
}

# Set up page configurations with wide widescreen proportions
st.set_page_config(page_title="Subcontinental AQI Engine", page_icon="🌍", layout="wide")

# Custom CSS injected directly to make the UI look premium, dark, and sleek
st.markdown("""
    <style>
        /* Hide default Streamlit padding and headers */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        div[data-testid="stDecoration"] { display: none; }
        
        /* Make standard tables look premium and clean */
        .stDataFrame div { border-radius: 10px; }
        
        /* Smooth button transitions */
        button[kind="primary"] {
            background-color: #2979FF !important;
            border: none !important;
            transition: all 0.3s ease;
        }
        button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(41, 121, 255, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

def call_api(method: str, path: str, **kwargs):
    try:
        resp = getattr(requests, method)(f"{API_BASE}{path}", timeout=15, **kwargs)
        if resp.status_code == 404: return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def main():
    # --- HEADER ARCHITECTURE ---
    st.markdown("<h1 style='text-align: center; margin-bottom: 5px; font-weight: 800;'>🌤️ Pan-India AQI Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9E9E9E; font-size: 16px; margin-bottom: 30px;'>Continuous Deep-Learning atmospheric telemetry mapping across the Indian subcontinent</p>", unsafe_allow_html=True)

    # --- SEARCH INTERFACE ---
    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
    with search_col2:
        with st.container():
            location = st.text_input(
                "Enter Location Name:", 
                placeholder="Search any city, town, or district in India (e.g. Satna, Pune, Bhopal)...", 
                label_visibility="collapsed"
            )
            
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                fetch_btn = st.button("🚀 Analyze Atmospheric Profile", use_container_width=True, type="primary")
            with btn_col2:
                # Fixed background sync button with completely hidden parameters
                sync_btn = st.button("🔄 Sync Grid", use_container_width=True, help="Trigger background incremental update across all 1,153 PostGIS grid points.")

    # Handle hidden background data sync with hardcoded optimized production values
    if sync_btn:
        with st.spinner("Deploying asynchronous tracking matrix across 1,153 grid nodes..."):
            # Resolution is fixed to 0.5° (~55km granularity), history depth is fixed to 3 days
            response = call_api("post", "/ingest/pan-india?resolution=0.50&lookback_days=3")
            if response and response.get("status") == "accepted":
                st.toast("⚡ Grid update worker array deployed successfully in background container context!", icon="✅")
            else:
                st.error("❌ Critical: Backend API refused the grid sync instruction sequence.")

    # Handle location search queries
    if fetch_btn and location.strip():
        with st.spinner(f"Resolving coordinates and querying spatial models for '{location.strip()}'..."):
            try:
                geo = geocode(location.strip())
                st.session_state["lat"] = geo.lat
                st.session_state["lon"] = geo.lon
                st.session_state["loc_name"] = geo.display_name
                time.sleep(0.5)
                st.rerun()
            except LocationOutsideIndiaError:
                st.error("🗺️ Boundary Restriction: Please select an analytical target point located inside India.")
            except GeocodingError:
                st.error("🔍 Location Unresolved: Could not calculate coordinates. Try adding the specific state descriptor name.")
            except Exception as e:
                st.error(f"❌ Handshake Failure: {e}")

    # --- REAL-TIME RESULTS DASHBOARD ---
    lat = st.session_state.get("lat")
    lon = st.session_state.get("lon")
    
    if not lat or not lon:
        st.markdown("<div style='text-align: center; margin-top: 80px; color: #616161; font-size: 15px;'>✨ Waiting for location query sequence input profile above...</div>", unsafe_allow_html=True)
        return

    loc_name = st.session_state.get("loc_name", "Selected Node Target")
    st.markdown(f"---")
    st.markdown(f"<h2 style='text-align: center; font-weight: 700; margin-bottom: 20px;'>📍 {loc_name}</h2>", unsafe_allow_html=True)
    
    # Elegant Full-Width Map Layout
    map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_df, size=40, zoom=10)
    st.write("")

    # Fetch point arrays via coordinates computed by the background geocoder processing unit
    aqi_data = call_api("get", f"/aqi?lat={lat}&lon={lon}")
    forecast_data = call_api("get", f"/forecast?lat={lat}&lon={lon}")

    if not aqi_data or not forecast_data:
        st.info("⏳ Spatial interpolation running in container context... If your database cache is currently cold, click the 'Sync Grid' button to backfill this coordinate region block.")
        return

    # Modern Two-Column Layout Panel Setup
    c1, c2 = st.columns([1, 2], gap="large")
    
    with c1:
        st.markdown("<h4 style='font-weight:600; margin-bottom:15px;'>Live Air Quality Index</h4>", unsafe_allow_html=True)
        aqi = aqi_data["aqi"]
        cat = aqi_data["aqi_category"]
        color = CATEGORY_COLOURS.get(cat, "#757575")
        
        # Premium Custom Neon Floating Dashboard Card
        st.markdown(
            f"""
            <div style="background: linear-gradient(145deg, #1A1A1A, #121212); padding: 35px; border-radius: 18px; text-align: center; border-left: 6px solid {color}; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                <h1 style="font-size: 85px; margin: 0; font-weight: 900; color: #FFFFFF; line-height: 1;">{aqi}</h1>
                <div style="background-color: {color}20; display: inline-block; padding: 6px 16px; border-radius: 20px; margin-top: 15px;">
                    <span style="color: {color}; font-weight: 700; font-size: 14px; letter-spacing: 1px;">{cat.upper()}</span>
                </div>
                <p style="color: #B0BEC5; font-size: 14px; margin-top: 20px; margin-bottom: 0;">Prominent Carrier Pollutant: <b style="color: #FFF;">{aqi_data['prominent_pollutant']}</b></p>
                <p style="color: #546E7A; font-size: 11px; margin-top: 8px; margin-bottom: 0;">Resolved via Nearest Grid Node Intersect: {aqi_data['location_name']}</p>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("<br><h4 style='font-weight:600; margin-bottom:10px;'>Atmospheric Telemetry Concentrations</h4>", unsafe_allow_html=True)
        p = aqi_data.get("pollutants", {})
        df_p = pd.DataFrame({
            "Pollutant Vector": ["PM2.5 (24h Metric)", "PM10 (24h Metric)", "NO₂ (Gas Tracer)", "O₃ (8h Bound)", "SO₂ (Gas Tracer)"],
            "Measured Concentration": [
                f"{p.get('pm25_24h_avg', 0):.1f} µg/m³",
                f"{p.get('pm10_24h_avg', 0):.1f} µg/m³",
                f"{p.get('no2_24h_avg', 0):.1f} µg/m³",
                f"{p.get('o3_8h_max', 0):.1f} µg/m³",
                f"{p.get('so2_24h_avg', 0):.1f} µg/m³",
            ]
        })
        st.dataframe(df_p, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("<h4 style='font-weight:600; margin-bottom:15px;'>24-Hour Predictive Horizon Trend (CPCB Scale)</h4>", unsafe_allow_html=True)
        hourly = forecast_data["hourly"]
        
        times = [h["forecast_target_time"][11:16] for h in hourly] 
        ensembles = [h["aqi_forecast"] for h in hourly]
        
        # High-End Plotly Visualization Setup
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=ensembles, mode='lines+markers',
            line=dict(color='#2979FF', width=4, shape='spline'),
            marker=dict(size=8, color='#FFFFFF', line=dict(color='#2979FF', width=2.5)),
            fill='tozeroy', fillcolor='rgba(41, 121, 255, 0.06)',
            name="Ensemble Forecasting Vector"
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, title="Timeline Horizon Hour (UTC)", titlefont=dict(color='#757575', size=12)),
            yaxis=dict(showgrid=True, gridcolor='#262626', title="CPCB Index Scale Value", titlefont=dict(color='#757575', size=12)),
            height=390,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        alpha = forecast_data.get("ensemble_alpha", 0.6)
        st.caption(f"🧠 Core Engine: Spatially-Aware Deep Learning Stack (LSTM Core: {alpha:.0%} Weight Assignment | XGBoost Regressor Residuals: {1-alpha:.0%})")

if __name__ == "__main__":
    main()