"""
Streamlit dashboard for Part 2 AQI MLOps.

Pages:
1. 🏠 Home / Overview: city selector, current AQI status
2. 📈 Historical Trend: AQI time series with feature overlays
3. 🔮 Next-Day Prediction: call FastAPI /predict, show result with color
4. 🤖 Model Comparison: load reports/model_comparison.csv
5. 📊 Monitoring Panel: drift + rolling error from ml.monitoring_metrics

Run: streamlit run app/streamlit_app.py
     or: docker compose up streamlit
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta

# Ensure project root is on path when running from app/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Analytics & Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour map for AQI categories ─────────────────────────────────────────────
AQI_COLORS = {
    "Good":                              "#00e400",
    "Moderate":                          "#ffff00",
    "Unhealthy for Sensitive Groups":    "#ff7e00",
    "Unhealthy":                         "#ff0000",
    "Very Unhealthy":                    "#8f3f97",
    "Hazardous":                         "#7e0023",
    "Unknown":                           "#cccccc",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db_engine():
    from src.database.connection import get_engine
    return get_engine()


@st.cache_data(ttl=300)
def load_aqi_history(city: str):
    """Load analytics.daily_aqi for the given city (cached 5 min)."""
    engine = get_db_engine()
    query = text("""
        SELECT date, city, aqi_value, aqi_category, avg_pm25, avg_pm10,
               avg_temperature, avg_humidity, avg_wind_speed, total_precipitation
        FROM analytics.daily_aqi
        WHERE city = :city
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"city": city}, parse_dates=["date"])
    return df


@st.cache_data(ttl=300)
def load_monitoring_metrics():
    """Load recent monitoring results from ml.monitoring_metrics."""
    engine = get_db_engine()
    try:
        query = text("""
            SELECT checked_at, missing_rate, stale_data_flag,
                   n_drifted_features, drift_triggered,
                   rolling_rmse, rolling_mae, baseline_rmse, rmse_pct_change,
                   rolling_f1, retrain_triggered, retrain_reason
            FROM ml.monitoring_metrics
            ORDER BY checked_at DESC
            LIMIT 30
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn, parse_dates=["checked_at"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_model_comparison():
    """Load model_comparison.csv from reports/."""
    path = PROJECT_ROOT / "reports" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def get_prediction(city: str, target_date: str):
    """Call the FastAPI /predict endpoint."""
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    try:
        resp = requests.post(
            f"{api_url}/predict",
            json={"city": city, "target_date": target_date},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"API error {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is `make serve` running?"
    except Exception as e:
        return None, str(e)


def get_api_health():
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unreachable", "models_loaded": False}


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("AQI Intelligence")
st.sidebar.markdown("**Production MLOps Platform**")

TARGET_CITY = os.environ.get("TARGET_CITY", "Bengaluru")
city = st.sidebar.selectbox("City", [TARGET_CITY], index=0)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Historical Trends", "Inference & Forecasting", "Model Benchmark", "Drift & Monitoring"],
)

# API health badge in sidebar
health = get_api_health()
status_label = "ONLINE" if health.get("status") == "ok" else ("DEGRADED" if health.get("status") == "degraded" else "OFFLINE")
st.sidebar.markdown(f"**Service Status**: `{status_label}`")
if health.get("regressor_version"):
    st.sidebar.markdown(f"Regressor: `v{health['regressor_version']}` | Classifier: `v{health['classifier_version']}`")


# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "Overview":
    st.title(f"Air Quality Intelligence Overview — {city}")

    try:
        hist = load_aqi_history(city)
    except Exception as e:
        st.error(f"Cannot load data: {e}. Have you run `make backfill`?")
        st.stop()

    if hist.empty:
        st.warning("No data found. Run `make backfill` to load historical data.")
        st.stop()

    latest = hist.iloc[-1]
    cat = latest.get("aqi_category", "Unknown")
    color = AQI_COLORS.get(cat, "#cccccc")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest AQI", f"{latest.get('aqi_value', 'N/A'):.0f}", help="Most recent daily AQI")
    col2.metric("Category", cat)
    col3.metric("Avg PM2.5", f"{latest.get('avg_pm25', 'N/A'):.1f} µg/m³")
    col4.metric("Data Date", str(latest["date"].date()))

    st.markdown(f"<div style='background:{color};padding:8px;border-radius:4px;color:#000'>"
                f"<b>Current AQI Category: {cat}</b></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total days tracked", len(hist))
    c2.metric("Average AQI", f"{hist['aqi_value'].mean():.1f}")
    c3.metric("Worst AQI day", f"{hist['aqi_value'].max():.0f} ({hist.loc[hist['aqi_value'].idxmax(), 'date'].date()})")


# ── Page: Historical Trend ────────────────────────────────────────────────────

elif page == "Historical Trends":
    st.title(f"Historical Observation Trends — {city}")

    try:
        hist = load_aqi_history(city)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if hist.empty:
        st.warning("No data. Run `make backfill`.")
        st.stop()

    # Date range filter
    min_date = hist["date"].min().date()
    max_date = hist["date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(max_date - timedelta(days=90), max_date),
        min_value=min_date, max_value=max_date,
    )
    if len(date_range) == 2:
        mask = (hist["date"].dt.date >= date_range[0]) & (hist["date"].dt.date <= date_range[1])
        hist = hist[mask]

    # AQI time series with category colour
    fig = px.area(
        hist, x="date", y="aqi_value",
        color_discrete_sequence=["steelblue"],
        title="Daily AQI",
        labels={"aqi_value": "AQI", "date": "Date"},
    )
    # Add category colour band
    for cat, color in AQI_COLORS.items():
        cat_rows = hist[hist["aqi_category"] == cat]
        if not cat_rows.empty:
            fig.add_scatter(
                x=cat_rows["date"], y=cat_rows["aqi_value"],
                mode="markers", name=cat,
                marker=dict(color=color, size=6),
            )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.line(hist, x="date", y=["avg_pm25", "avg_pm10"],
                       title="PM2.5 and PM10 Trends",
                       labels={"value": "µg/m³", "variable": "Pollutant"})
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.line(hist, x="date", y=["avg_temperature", "avg_humidity"],
                       title="Weather Trends",
                       labels={"value": "Value", "variable": "Metric"})
        st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(hist.tail(10), use_container_width=True)
    st.download_button(
        "Export Historical CSV",
        hist.to_csv(index=False),
        file_name=f"aqi_history_{city}.csv",
    )


# ── Page: Prediction ──────────────────────────────────────────────────────────

elif page == "Inference & Forecasting":
    st.title(f"Next-Day AQI Forecast Service — {city}")

    target_date = st.date_input(
        "Forecast target date",
        value=date.today() + timedelta(days=1),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=7),
    )

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Executing inference pipeline …"):
            result, error = get_prediction(city, target_date.isoformat())

        if error:
            st.error(error)
        else:
            cat = result.get("predicted_category", "Unknown")
            color = AQI_COLORS.get(cat, "#cccccc")

            st.markdown(f"<div style='background:{color};padding:16px;border-radius:8px;font-size:1.2rem'>"
                        f"<b>Predicted AQI for {target_date}: {result.get('predicted_aqi', 'N/A'):.0f} — {cat}</b>"
                        f"</div>", unsafe_allow_html=True)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted AQI",      f"{result.get('predicted_aqi', 'N/A'):.1f}")
            col2.metric("Predicted Category", result.get("predicted_category", "N/A"))
            col3.metric("Confidence",         result.get("aqi_confidence", "N/A"))

            with st.expander("Model details"):
                st.json({
                    "regressor_version":  result.get("regressor_version"),
                    "classifier_version": result.get("classifier_version"),
                    "timestamp":          result.get("timestamp"),
                    "warning":            result.get("warning"),
                })

    st.markdown("---")
    st.markdown("#### Recent Predictions")
    engine = get_db_engine()
    try:
        preds = pd.read_sql(
            text("""
                SELECT requested_at, target_date, predicted_aqi, predicted_category,
                       actual_aqi, regressor_version, latency_ms, status
                FROM ml.prediction_log
                WHERE city = :city ORDER BY requested_at DESC LIMIT 20
            """),
            engine.connect(), params={"city": city}, parse_dates=["requested_at", "target_date"]
        )
        st.dataframe(preds, use_container_width=True)
    except Exception:
        st.info("No prediction history yet.")


# ── Page: Model Comparison ────────────────────────────────────────────────────

elif page == "Model Benchmark":
    st.title("Model Performance Benchmarks")

    comp = load_model_comparison()
    if comp.empty:
        st.warning("No model comparison data. Run `make train` first.")
    else:
        st.markdown("### Regression Models")
        reg = comp[comp["type"] == "regression"].sort_values("rmse")
        st.dataframe(reg[["model", "split", "rmse", "mae", "r2"]].style.highlight_min(
            subset=["rmse", "mae"], color="#c6efce"
        ), use_container_width=True)

        fig = px.bar(
            reg[reg["split"] == "test"].sort_values("rmse"),
            x="model", y="rmse", color="model",
            title="Test RMSE by Model (lower is better)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Classification Models")
        clf = comp[comp["type"] == "classification"].sort_values("macro_f1", ascending=False)
        st.dataframe(clf[["model", "split", "accuracy", "macro_f1"]].style.highlight_max(
            subset=["accuracy", "macro_f1"], color="#c6efce"
        ), use_container_width=True)

    # Show saved figures
    figs_dir = PROJECT_ROOT / "reports" / "figures"
    fig_files = sorted(figs_dir.glob("*.png"))
    if fig_files:
        st.markdown("### Training Plots")
        cols = st.columns(2)
        for i, f in enumerate(fig_files[:6]):
            cols[i % 2].image(str(f), caption=f.stem, use_column_width=True)


# ── Page: Monitoring ──────────────────────────────────────────────────────────

elif page == "Drift & Monitoring":
    st.title("Data Quality & Feature Drift Telemetry")

    mon = load_monitoring_metrics()
    if mon.empty:
        st.warning("No monitoring data yet. Run `make monitor`.")
    else:
        latest_mon = mon.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Missing Rate",       f"{(latest_mon.get('missing_rate') or 0):.1%}")
        col2.metric("Drifted Features",   int(latest_mon.get("n_drifted_features") or 0))
        col3.metric("Rolling RMSE",       f"{(latest_mon.get('rolling_rmse') or 'N/A')}")
        col4.metric("Retrain Triggered",  str(latest_mon.get("retrain_triggered")))

        if latest_mon.get("retrain_triggered"):
            st.error(f"Retraining Trigger Active: {latest_mon.get('retrain_reason')}")

        st.markdown("---")
        st.markdown("### Rolling RMSE Over Time")
        if "rolling_rmse" in mon.columns and mon["rolling_rmse"].notna().any():
            fig = px.line(mon.sort_values("checked_at"), x="checked_at", y="rolling_rmse",
                          title="Rolling RMSE", labels={"rolling_rmse": "RMSE"})
            if "baseline_rmse" in mon.columns and mon["baseline_rmse"].notna().any():
                fig.add_hline(
                    y=mon["baseline_rmse"].dropna().iloc[0],
                    line_dash="dash", annotation_text="Baseline RMSE",
                )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Drifted Features Over Time")
        if "n_drifted_features" in mon.columns:
            fig2 = px.bar(mon.sort_values("checked_at"), x="checked_at", y="n_drifted_features",
                          title="Number of Drifted Features",
                          color="drift_triggered",
                          color_discrete_map={True: "red", False: "steelblue"})
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Monitoring History")
        st.dataframe(mon[["checked_at", "missing_rate", "n_drifted_features",
                           "rolling_rmse", "rolling_f1", "retrain_triggered"]].head(10),
                     use_container_width=True)

    # Export for Tableau
    if not mon.empty:
        tableau_dir = PROJECT_ROOT / "tableau_exports"
        tableau_dir.mkdir(exist_ok=True)
        mon_path = tableau_dir / "monitoring_export.csv"
        mon.to_csv(mon_path, index=False)
        st.success(f"Monitoring data exported to {mon_path}")

    # Read monitoring report markdown
    report_path = PROJECT_ROOT / "reports" / "monitoring_report.md"
    if report_path.exists():
        with st.expander("Inspection: Detailed Monitoring Report (Markdown)"):
            st.markdown(report_path.read_text())
