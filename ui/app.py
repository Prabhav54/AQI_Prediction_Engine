"""
ui/app.py
---------
Premium, High-Performance Streamlit Frontend for the Pan-India AQI Prediction Engine.
Features an ultra-clean single-page interface with hardcoded, production-optimized background values.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import geocoder and safety modules from your core engine layers
from ingestion.geocoder import geocode
from exceptions import LocationOutsideIndiaError, GeocodingError

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

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
        resp = getattr(requests, method)(f"{API_BASE}{path}", timeout=30, **kwargs)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_nodes():
    return call_api("get", "/nodes") or []


def render_map(nodes, selected_hash=None):
    df = pd.DataFrame(nodes)
    df["size"] = np.where(df["kind"] == "city", 9, 6)
    fig = go.Figure()
    for cat, colour in CATEGORY_COLOURS.items():
        d = df[df["aqi_category"] == cat]
        if d.empty:
            continue
        fig.add_trace(go.Scattermap(
            lat=d["lat"], lon=d["lon"], mode="markers", name=cat,
            marker=dict(size=d["size"], color=colour, opacity=0.9),
            text=d["location_name"] + "<br>AQI " + d["aqi"].astype(str) + " · " + d["prominent_pollutant"],
            hoverinfo="text",
        ))
    if selected_hash is not None:
        s = df[df["location_hash"] == selected_hash]
        fig.add_trace(go.Scattermap(lat=s["lat"], lon=s["lon"], mode="markers", showlegend=False,
                                    marker=dict(size=18, color="#FFFFFF", opacity=0.35), hoverinfo="skip"))
    fig.update_layout(
        map=dict(style="carto-darkmatter", center=dict(lat=22.5, lon=80), zoom=3.6),
        margin=dict(l=0, r=0, t=0, b=0), height=470, paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(18,18,18,0.7)",
                    font=dict(color="#E0E0E0")),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _fmt(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def main():
    # --- HEADER ARCHITECTURE ---
    st.markdown("<h1 style='text-align: center; margin-bottom: 5px; font-weight: 800;'>"
                "🌤️ Pan-India AQI Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9E9E9E; font-size: 16px; margin-bottom: 30px;'>"
                "24-hour CPCB AQI and pollutant forecasts for every part of India · "
                "LSTM + GRU + LightGBM ensemble</p>", unsafe_allow_html=True)

    nodes = load_nodes()
    if not nodes:
        st.info("⏳ No data yet. Start the API and click **Sync Grid**, "
                "or run `python run_pipeline.py all`.")

    # --- SEARCH INTERFACE ---
    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
    with search_col2:
        cities = sorted([n for n in nodes if n["kind"] == "city"], key=lambda n: n["location_name"])
        names = ["—"] + [f"{n['location_name']}, {n['state']}" for n in cities]
        picked = st.selectbox("Tracked city", names, label_visibility="collapsed")
        location = st.text_input(
            "Or search any place:",
            placeholder="…or search any town or district (e.g. Satna, Bhopal) — nearest node is used",
            label_visibility="collapsed",
        )
        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            fetch_btn = st.button("🚀 Analyze Atmospheric Profile", use_container_width=True, type="primary")
        with btn_col2:
            sync_btn = st.button("🔄 Sync Grid", use_container_width=True,
                                 help="Run ingest → forecast for every pan-India node (a few minutes).")

    if sync_btn:
        response = call_api("post", "/ingest/pan-india?resolution=1.0")
        if response and response.get("status") == "accepted":
            st.toast("⚡ Pan-India pipeline started in the background.", icon="✅")
        else:
            st.error("❌ Backend API refused the sync request.")

    if picked != "—":
        node = cities[names.index(picked) - 1]
        st.session_state.update(lat=node["lat"], lon=node["lon"], loc_name=node["location_name"])

    if fetch_btn and location.strip():
        with st.spinner(f"Resolving '{location.strip()}'..."):
            try:
                geo = geocode(location.strip())
                st.session_state.update(lat=geo.lat, lon=geo.lon, loc_name=geo.display_name)
            except LocationOutsideIndiaError:
                st.error("🗺️ Please choose a location inside India.")
            except GeocodingError:
                st.error("🔍 Could not resolve that place. Try adding the state name.")
            except Exception as e:
                st.error(f"❌ Geocoding failed: {e}")

    lat, lon = st.session_state.get("lat"), st.session_state.get("lon")
    aqi_data = call_api("get", f"/aqi?lat={lat}&lon={lon}") if lat else None
    if nodes:
        render_map(nodes, aqi_data["node"]["location_hash"] if aqi_data else None)

    if not lat:
        st.markdown("<div style='text-align: center; margin-top: 20px; color: #616161; font-size: 15px;'>"
                    "✨ Pick a city or search a place to see its 24-hour forecast.</div>",
                    unsafe_allow_html=True)
        return

    forecast_data = call_api("get", f"/forecast?lat={lat}&lon={lon}")
    if not aqi_data or not forecast_data:
        st.info("⏳ No data for the nearest node yet — click 'Sync Grid' to run the pipeline.")
        return

    node = aqi_data["node"]
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center; font-weight: 700; margin-bottom: 4px;'>📍 "
                f"{st.session_state.get('loc_name', node['location_name'])}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #78909C; margin-top: 0;'>Nearest node: "
                f"{node['location_name']} · {node['distance_km']:.0f} km away</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.markdown("<h4 style='font-weight:600; margin-bottom:15px;'>Live Air Quality Index</h4>",
                    unsafe_allow_html=True)
        aqi = aqi_data["aqi"]
        cat = aqi_data["aqi_category"]
        color = CATEGORY_COLOURS.get(cat, "#757575")
        st.markdown(
            f"""
            <div style="background: linear-gradient(145deg, #1A1A1A, #121212); padding: 35px; border-radius: 18px; text-align: center; border-left: 6px solid {color}; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                <h1 style="font-size: 85px; margin: 0; font-weight: 900; color: #FFFFFF; line-height: 1;">{aqi}</h1>
                <div style="background-color: {color}20; display: inline-block; padding: 6px 16px; border-radius: 20px; margin-top: 15px;">
                    <span style="color: {color}; font-weight: 700; font-size: 14px; letter-spacing: 1px;">{cat.upper()}</span>
                </div>
                <p style="color: #B0BEC5; font-size: 14px; margin-top: 20px; margin-bottom: 0;">Prominent pollutant: <b style="color: #FFF;">{aqi_data['prominent_pollutant']}</b></p>
                <p style="color: #546E7A; font-size: 11px; margin-top: 8px; margin-bottom: 0;">CPCB AQI · {aqi_data['timestamp'][:16].replace('T', ' ')} UTC</p>
            </div>
            """, unsafe_allow_html=True  # noqa: E501
        )

        st.markdown("<br><h4 style='font-weight:600; margin-bottom:10px;'>Current Concentrations</h4>",
                    unsafe_allow_html=True)
        p = aqi_data.get("pollutants", {})
        df_p = pd.DataFrame({
            "Pollutant": ["PM2.5 (24h avg)", "PM10 (24h avg)", "NO₂ (24h avg)",
                          "SO₂ (24h avg)", "CO (8h avg)", "O₃ (8h avg)"],
            "Concentration": [
                f"{_fmt(p.get('pm25_24h_avg'))} µg/m³", f"{_fmt(p.get('pm10_24h_avg'))} µg/m³",
                f"{_fmt(p.get('no2_24h_avg'))} µg/m³", f"{_fmt(p.get('so2_24h_avg'))} µg/m³",
                f"{_fmt(p.get('co_8h_avg'), 2)} mg/m³", f"{_fmt(p.get('o3_8h_avg'))} µg/m³",
            ],
            "Sub-index": [_fmt(p.get(f"sub_index_{k}"), 0) for k in ("pm25", "pm10", "no2", "so2", "co", "o3")],
        })
        st.dataframe(df_p, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("<h4 style='font-weight:600; margin-bottom:15px;'>24-Hour Forecast "
                    "(CPCB AQI, 80% interval)</h4>", unsafe_allow_html=True)
        hourly = pd.DataFrame(forecast_data["hourly"])
        hourly["t"] = pd.to_datetime(hourly["forecast_target_time"]).dt.tz_convert("Asia/Kolkata")

        fig = go.Figure()
        if hourly["aqi_upper"].notna().any():
            fig.add_trace(go.Scatter(x=hourly["t"], y=hourly["aqi_upper"], mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hourly["t"], y=hourly["aqi_lower"], mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor="rgba(41, 121, 255, 0.15)", name="80% interval"))
        fig.add_trace(go.Scatter(
            x=hourly["t"], y=hourly["aqi"], mode="lines+markers",
            line=dict(color="#2979FF", width=4, shape="spline"),
            marker=dict(size=7, color="#FFFFFF", line=dict(color="#2979FF", width=2.5)),
            name="Ensemble forecast",
        ))
        for member, colour in (("aqi_lstm", "#00E676"), ("aqi_gru", "#FFEA00"), ("aqi_lgbm", "#FF9100")):
            fig.add_trace(go.Scatter(x=hourly["t"], y=hourly[member], mode="lines", name=member[4:].upper(),
                                     line=dict(color=colour, width=1, dash="dot"), visible="legendonly"))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, title="Time (IST)"),
            yaxis=dict(showgrid=True, gridcolor="#262626", title="CPCB AQI"),
            height=390, hovermode="x unified", legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        outlook = hourly.set_index("hours_ahead").loc[[6, 12, 24], ["pm25", "pm10", "no2", "so2", "co", "o3"]]
        outlook.index = [f"+{h}h" for h in outlook.index]
        outlook.columns = ["PM2.5", "PM10", "NO₂", "SO₂", "CO (mg)", "O₃"]
        st.dataframe(outlook.round(1), use_container_width=True)
        st.caption(f"🧠 Model {forecast_data.get('model_version')} · issued "
                   f"{forecast_data['generated_at'][:16].replace('T', ' ')} UTC · peak {forecast_data['aqi_max']} "
                   f"· mostly {forecast_data['dominant_category']}")


if __name__ == "__main__":
    main()
