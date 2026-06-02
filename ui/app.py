"""
ui/app.py
---------
Module 5 — Streamlit Frontend

This is the user-facing dashboard. It talks to the FastAPI backend
and shows three things:
  1. A search bar where you type any Indian city
  2. A real-time AQI gauge with the prominent pollutant
  3. A 24-hour forecast line chart with ensemble breakdown

The app doesn't do any computation — it just calls the API and
renders what comes back. All the heavy work stays in the backend.

Run with:
  streamlit run ui/app.py

Make sure the FastAPI server is running first:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import aqi_category

# ── Config ──────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# CPCB category colours — used consistently across all charts
CATEGORY_COLOURS = {
    "Good":         "#00b050",
    "Satisfactory": "#92d050",
    "Moderate":     "#ffff00",
    "Poor":         "#ff9900",
    "Very Poor":    "#ff0000",
    "Severe":       "#c00000",
}

# ── Page setup ──────────────────────────────────────────────────
st.set_page_config(
    page_title = "Pan-India AQ Engine",
    page_icon  = "🌫️",
    layout     = "wide",
)


# ================================================================
# Helper: call the API
# ================================================================

def call_api(method: str, path: str, **kwargs) -> dict | None:
    """
    Thin wrapper around requests so error handling is consistent.
    Returns the JSON response dict, or None if the call failed.
    Shows a Streamlit error message on failure.
    """
    try:
        resp = getattr(requests, method)(f"{API_BASE}{path}", timeout=60, **kwargs)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API server. "
            "Start it with: `uvicorn api.main:app --reload`"
        )
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.json().get('detail', str(exc))}")
        return None


# ================================================================
# Helper: AQI gauge (Plotly)
# ================================================================

def aqi_gauge(aqi: int, category: str, prominent: str) -> go.Figure:
    """
    Circular gauge showing the AQI value with colour-coded needle.
    The colour matches the CPCB category.
    """
    colour = CATEGORY_COLOURS.get(category, "#888888")

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = aqi,
        title = {
            "text": f"<b>AQI</b><br><span style='font-size:14px'>{prominent} is the prominent pollutant</span>",
            "font": {"size": 18},
        },
        number = {"font": {"size": 60, "color": colour}},
        gauge  = {
            "axis":  {"range": [0, 500], "tickwidth": 1},
            "bar":   {"color": colour, "thickness": 0.3},
            "steps": [
                {"range": [0,   50],  "color": "#e8f5e9"},
                {"range": [50,  100], "color": "#f9fbe7"},
                {"range": [100, 200], "color": "#fffde7"},
                {"range": [200, 300], "color": "#fff3e0"},
                {"range": [300, 400], "color": "#fce4ec"},
                {"range": [400, 500], "color": "#f3e5f5"},
            ],
            "threshold": {
                "line":  {"color": colour, "width": 4},
                "thickness": 0.75,
                "value": aqi,
            },
        },
    ))

    fig.update_layout(height=320, margin=dict(t=60, b=20, l=30, r=30))
    return fig


# ================================================================
# Helper: 24-hour forecast chart (Plotly)
# ================================================================

def forecast_chart(hourly: list[dict], alpha: float) -> go.Figure:
    """
    Line chart showing the 24-hour AQI forecast.
    Plots ensemble blend + individual LSTM and XGBoost predictions
    so you can see how much each model contributes.
    """
    times      = [h["forecast_target_time"] for h in hourly]
    ensemble   = [h["aqi_forecast"]          for h in hourly]
    lstm_vals  = [h.get("aqi_lstm", None)    for h in hourly]
    xgb_vals   = [h.get("aqi_xgb",  None)   for h in hourly]

    fig = go.Figure()

    # Individual model predictions — lighter, dashed
    if any(v is not None for v in lstm_vals):
        fig.add_trace(go.Scatter(
            x=times, y=lstm_vals,
            name=f"LSTM ({alpha:.0%})",
            line=dict(color="#7986cb", dash="dot", width=1.5),
            opacity=0.7,
        ))

    if any(v is not None for v in xgb_vals):
        fig.add_trace(go.Scatter(
            x=times, y=xgb_vals,
            name=f"XGBoost ({1-alpha:.0%})",
            line=dict(color="#ef9a9a", dash="dot", width=1.5),
            opacity=0.7,
        ))

    # Ensemble blend — bold, prominent
    fig.add_trace(go.Scatter(
        x=times, y=ensemble,
        name="Ensemble Forecast",
        line=dict(color="#1565c0", width=3),
        mode="lines+markers",
        marker=dict(size=5),
    ))

    # Colour bands for AQI categories
    category_bands = [
        (0,   50,  CATEGORY_COLOURS["Good"],         "Good"),
        (50,  100, CATEGORY_COLOURS["Satisfactory"],  "Satisfactory"),
        (100, 200, CATEGORY_COLOURS["Moderate"],      "Moderate"),
        (200, 300, CATEGORY_COLOURS["Poor"],          "Poor"),
        (300, 400, CATEGORY_COLOURS["Very Poor"],     "Very Poor"),
        (400, 500, CATEGORY_COLOURS["Severe"],        "Severe"),
    ]

    for lo, hi, colour, label in category_bands:
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor=colour, opacity=0.08,
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=9,
        )

    fig.update_layout(
        title      = "24-Hour AQI Forecast (Ensemble LSTM + XGBoost)",
        xaxis_title= "Time",
        yaxis_title= "AQI",
        yaxis      = dict(range=[0, max(max(ensemble) + 50, 200)]),
        legend     = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height     = 420,
        margin     = dict(t=60, b=40, l=50, r=80),
        plot_bgcolor = "white",
    )

    return fig


# ================================================================
# Helper: Pollutant breakdown bar chart
# ================================================================

def pollutant_chart(pollutants: dict) -> go.Figure:
    """
    Horizontal bar chart of individual CPCB sub-indices.
    Makes it easy to see which pollutant is closest to a higher category.
    """
    sub_index_cols = {
        "PM2.5": pollutants.get("sub_index_pm25"),
        "PM10":  pollutants.get("sub_index_pm10"),
        "NO₂":   pollutants.get("sub_index_no2"),
        "SO₂":   pollutants.get("sub_index_so2"),
        "CO":    pollutants.get("sub_index_co"),
        "O₃":    pollutants.get("sub_index_o3"),
    }

    labels = [k for k, v in sub_index_cols.items() if v is not None]
    values = [v for v in sub_index_cols.values()    if v is not None]
    colours = [
        CATEGORY_COLOURS.get(aqi_category(int(v)), "#888888")
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title       = "Pollutant Sub-Indices (CPCB)",
        xaxis_title = "Sub-Index (0–500)",
        xaxis       = dict(range=[0, 520]),
        height      = 280,
        margin      = dict(t=50, b=30, l=60, r=60),
        plot_bgcolor= "white",
    )

    return fig


# ================================================================
# Main app layout
# ================================================================

def main():
    # Header
    st.title("🌫️ Pan-India Air Quality Engine")
    st.caption(
        "Real-time CPCB AQI · Satellite + Weather Fusion · "
        "LSTM + XGBoost Ensemble Forecast"
    )

    # API health check banner
    health = call_api("get", "/health")
    if health:
        db_ok     = health.get("db") == "ok"
        models_ok = health.get("models") == "ready"
        col_h1, col_h2, col_h3 = st.columns(3)
        col_h1.metric("API",    "🟢 Online")
        col_h2.metric("Database", "🟢 Connected" if db_ok     else "🔴 Offline")
        col_h3.metric("Models",   "🟢 Ready"     if models_ok else "🟡 Training needed")

    st.divider()

    # ── Search bar ──────────────────────────────────────────────
    st.subheader("Search a city")
    col_search, col_mock, col_btn = st.columns([3, 1, 1])

    with col_search:
        location = st.text_input(
            label       = "Location",
            placeholder = "e.g.  Delhi, Mumbai, Rourkela",
            label_visibility = "collapsed",
        )

    with col_mock:
        use_mock = st.checkbox(
            "Mock satellite",
            value  = False,
            help   = "Use synthetic GEE data (no credentials needed). "
                     "Good for testing offline."
        )

    with col_btn:
        run_ingest = st.button("🔄  Fetch Data", use_container_width=True)

    # ── Trigger ingestion ────────────────────────────────────────
    if run_ingest and location.strip():
        with st.spinner(f"Pulling data for '{location}'... (may take 30–60 seconds)"):
            result = call_api("post", "/ingest/", json={
                "location":           location.strip(),
                "lookback_days":      7,
                "use_mock_satellite": use_mock,
            })

        if result:
            st.success(
                f"✅ Data pull queued for **{result.get('location', 'your chosen city')}**."
                "Refreshing in a moment..."
            )
            # Store resolved coords in session state so the display
            # sections below can use them
            st.session_state["lat"]           = result["lat"]
            st.session_state["lon"]           = result["lon"]
            st.session_state["resolved_name"] = result.get("location", "Unknown Location")
            # Give the background pipeline a few seconds before displaying
            import time; time.sleep(5)
            st.rerun()

    st.divider()

    # ── AQI display ─────────────────────────────────────────────
    lat = st.session_state.get("lat")
    lon = st.session_state.get("lon")

    if lat is None or lon is None:
        st.info(
            "👆 Enter a city name above and click **Fetch Data** to get started."
        )
        return

    resolved_name = st.session_state.get("resolved_name", f"{lat:.3f}, {lon:.3f}")
    st.subheader(f"📍 {resolved_name}")

    # Fetch current AQI
    aqi_data = call_api("get", f"/aqi?lat={lat}&lon={lon}")

    if aqi_data is None:
        st.warning(
            "Data is still being processed — "
            "click **Fetch Data** again in a few seconds."
        )
        return

    # ── Row 1: Gauge + Pollutant breakdown ──────────────────────
    col_gauge, col_pollutants = st.columns([1, 1.4])

    with col_gauge:
        fig_gauge = aqi_gauge(
            aqi        = aqi_data["aqi"],
            category   = aqi_data["aqi_category"],
            prominent  = aqi_data["prominent_pollutant"],
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Category badge
        cat   = aqi_data["aqi_category"]
        colour = CATEGORY_COLOURS.get(cat, "#888888")
        st.markdown(
            f"<div style='text-align:center; padding:8px; "
            f"background:{colour}22; border-radius:8px; "
            f"border:2px solid {colour}; font-weight:bold; font-size:18px'>"
            f"{cat}</div>",
            unsafe_allow_html=True,
        )

    with col_pollutants:
        fig_poll = pollutant_chart(aqi_data.get("pollutants", {}))
        st.plotly_chart(fig_poll, use_container_width=True)

        # Raw concentration table
        p = aqi_data.get("pollutants", {})
        st.caption("24h averages / 8h max")
        conc_data = {
            "Pollutant": ["PM2.5", "PM10", "NO₂",  "SO₂",  "CO",   "O₃"],
            "Value":     [
                f"{p.get('pm25_24h_avg', 'N/A'):.1f} µg/m³" if p.get("pm25_24h_avg") else "N/A",
                f"{p.get('pm10_24h_avg', 'N/A'):.1f} µg/m³" if p.get("pm10_24h_avg") else "N/A",
                f"{p.get('no2_24h_avg',  'N/A'):.1f} µg/m³" if p.get("no2_24h_avg")  else "N/A",
                f"{p.get('so2_24h_avg',  'N/A'):.1f} µg/m³" if p.get("so2_24h_avg")  else "N/A",
                f"{p.get('co_8h_max',    'N/A'):.2f} mg/m³"  if p.get("co_8h_max")   else "N/A",
                f"{p.get('o3_8h_max',    'N/A'):.1f} µg/m³"  if p.get("o3_8h_max")   else "N/A",
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(conc_data), hide_index=True, use_container_width=True)

    st.divider()

    # ── Row 2: 24-hour forecast chart ───────────────────────────
    st.subheader("📈 24-Hour Forecast")

    with st.spinner("Running ensemble forecast..."):
        forecast_data = call_api("get", f"/forecast?lat={lat}&lon={lon}")

    if forecast_data is None:
        st.warning(
            "Forecast not available yet — models may need training. "
            "Run: `python forecasting/train.py --location '{resolved_name}'`"
        )
        return

    alpha = forecast_data.get("ensemble_alpha", 0.60)

    # Ensemble weight info strip
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("LSTM Weight",    f"{alpha:.0%}")
    col_e2.metric("XGBoost Weight", f"{1-alpha:.0%}")
    col_e3.metric(
        "Dominant Category",
        forecast_data["dominant_category"],
        help="Most frequent AQI category over the next 24 hours"
    )

    fig_forecast = forecast_chart(forecast_data["hourly"], alpha)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # 24h summary table
    st.subheader("Hourly Breakdown")
    import pandas as pd
    df_forecast = pd.DataFrame([
        {
            "Hour":     f"T+{h['hours_ahead']}",
            "Time":     h["forecast_target_time"][:16].replace("T", " "),
            "AQI":      h["aqi_forecast"],
            "Category": h["aqi_category_forecast"],
            "LSTM":     h.get("aqi_lstm", "—"),
            "XGBoost":  h.get("aqi_xgb",  "—"),
        }
        for h in forecast_data["hourly"]
    ])
    st.dataframe(df_forecast, hide_index=True, use_container_width=True)

    # Footer
    st.divider()
    st.caption(
        "Data sources: Sentinel-5P (NO₂/SO₂/CO/O₃) · MODIS (AOD) · "
        "Open-Meteo (weather) · CPCB AQI breakpoints (Nov 2014)"
    )


if __name__ == "__main__":
    main()