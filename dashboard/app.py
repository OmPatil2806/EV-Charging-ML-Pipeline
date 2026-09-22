"""Streamlit dashboard: data exploration, model comparison leaderboard, and
a live prediction form served by the same model artifacts as the FastAPI
service (src/inference.py is the shared implementation for both).

Run with:
    streamlit run dashboard/app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import load_model_bundle, predict_session  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

# Validated categorical palette (dataviz skill reference palette, light mode)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DIVERGING_POS = "#2a78d6"
DIVERGING_NEG = "#e34948"
SEQUENTIAL = "#2a78d6"
SURFACE = "#fcfcfb"

VEHICLE_MODELS = ["Tesla Model 3", "Nissan Leaf", "Chevy Bolt", "BMW i3", "Hyundai Kona"]
CHARGER_TYPES = ["Level 1", "Level 2", "DC Fast Charger"]
USER_TYPES = ["Commuter", "Casual Driver", "Long-Distance Traveler"]

st.set_page_config(page_title="EV Charging Energy Prediction", layout="wide")

cfg = load_config()


@st.cache_data
def load_processed_data():
    path = resolve_path(cfg["data"]["processed_path"])
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_metrics():
    path = resolve_path(cfg["artifacts"]["dir"]) / cfg["artifacts"]["metrics_file"]
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def get_model_bundle():
    return load_model_bundle(cfg)


def style_fig(fig):
    fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, margin=dict(t=30, b=10))
    return fig


st.title("EV Charging Energy Prediction")
st.caption(
    "End-to-end ML pipeline: synthetic data with a real physical relationship -> "
    "cleaning -> feature engineering -> model comparison -> serving."
)

tab_eda, tab_models, tab_predict = st.tabs(["Data Exploration", "Model Comparison", "Live Prediction"])

# --------------------------------------------------------------------- EDA
with tab_eda:
    df = load_processed_data()
    if df is None:
        st.warning("No processed dataset found. Run `python pipeline/run_pipeline.py` first.")
    else:
        target = cfg["target_column"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Sessions", f"{len(df):,}")
        col2.metric("Features", df.shape[1] - 1)
        col3.metric("Avg. Energy Consumed", f"{df[target].mean():.1f} kWh")

        st.subheader("Feature distribution")
        numeric_cols = [c for c in cfg["numeric_features"] if c in df.columns]
        default_idx = numeric_cols.index("expected_energy_kwh") if "expected_energy_kwh" in numeric_cols else 0
        selected = st.selectbox("Numeric feature", numeric_cols, index=default_idx)
        fig = px.histogram(df, x=selected, nbins=40, color_discrete_sequence=[SEQUENTIAL])
        fig.update_layout(bargap=0.05)
        st.plotly_chart(style_fig(fig), width="stretch")

        st.subheader("Correlation with Energy Consumed")
        corr = df.corr(numeric_only=True)[target].drop(target).sort_values()
        colors = [DIVERGING_NEG if v < 0 else DIVERGING_POS for v in corr.values]
        fig2 = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color=colors))
        fig2.update_layout(xaxis_title="Correlation", yaxis_title="")
        st.plotly_chart(style_fig(fig2), width="stretch")

        st.subheader("Sessions by category")
        cat_cols = [c for c in cfg["categorical_features"] if c in df.columns]
        cat_selected = st.selectbox("Categorical feature", cat_cols)
        counts = df[cat_selected].value_counts().reset_index()
        counts.columns = [cat_selected, "count"]
        fig3 = px.bar(counts, x=cat_selected, y="count", color=cat_selected, color_discrete_sequence=CATEGORICAL)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig3), width="stretch")

# ------------------------------------------------------------- Model tab
with tab_models:
    metrics = load_metrics()
    if metrics is None:
        st.warning("No trained models found. Run `python pipeline/run_pipeline.py` first.")
    else:
        leaderboard = metrics["leaderboard"]
        best_name = metrics["best_model"]

        board_df = pd.DataFrame(
            [{"model": name, **values} for name, values in leaderboard.items()]
        ).sort_values("rmse")

        st.subheader("Test set leaderboard")
        st.dataframe(
            board_df.rename(
                columns={"model": "Model", "rmse": "RMSE (kWh)", "mae": "MAE (kWh)", "r2": "R2"}
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader("RMSE by model (lower is better)")
        fig = px.bar(board_df, x="model", y="rmse", color="model", color_discrete_sequence=CATEGORICAL)
        fig.update_layout(showlegend=False, yaxis_title="RMSE (kWh)", xaxis_title="")
        st.plotly_chart(style_fig(fig), width="stretch")

        st.success(f"Best model (lowest test RMSE): **{best_name}**")

        residual_path = resolve_path(cfg["artifacts"]["dir"]) / cfg["artifacts"]["residual_plot_file"]
        if residual_path.exists():
            st.subheader("Residuals (actual - predicted)")
            st.image(str(residual_path), width="stretch")

# --------------------------------------------------------- Live prediction
with tab_predict:
    metrics = load_metrics()
    if metrics is None:
        st.warning("No trained models found. Run `python pipeline/run_pipeline.py` first.")
    else:
        st.write("Enter a charging session's characteristics to predict energy consumed.")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                vehicle_model = st.selectbox("Vehicle model", VEHICLE_MODELS)
                battery_capacity_kwh = st.number_input("Battery capacity (kWh)", min_value=1.0, value=60.0)
                charger_type = st.selectbox("Charger type", CHARGER_TYPES, index=1)  # Level 2
                user_type = st.selectbox("User type", USER_TYPES)
            with c2:
                charge_date = st.date_input("Charging date", value=datetime.now().date())
                charge_time = st.time_input("Charging time", value=datetime.now().time())
                # defaults below are internally consistent: 60 kWh x 60% SoC gain / ~0.9
                # efficiency = ~40 kWh, at an 11 kW (Level 2) rate = ~3.6 hours
                charging_duration_hours = st.number_input("Charging duration (hours)", min_value=0.01, value=3.6)
                charging_rate_kw = st.number_input("Charging rate (kW)", min_value=0.1, value=11.0)
            with c3:
                charging_cost_usd = st.number_input("Charging cost (USD)", min_value=0.0, value=9.0)
                soc_start_pct = st.slider("SoC start (%)", 0, 100, 20)
                soc_end_pct = st.slider("SoC end (%)", 0, 100, 80)
                distance_since_last_charge_km = st.number_input(
                    "Distance since last charge (km)", min_value=0.0, value=150.0
                )

            col_age, col_temp = st.columns(2)
            with col_age:
                vehicle_age_years = st.number_input("Vehicle age (years)", min_value=0.0, value=2.0)
            with col_temp:
                temperature_c = st.number_input("Temperature (C)", value=20.0)

            submitted = st.form_submit_button("Predict energy consumed")

        if submitted:
            session = dict(
                vehicle_model=vehicle_model,
                battery_capacity_kwh=battery_capacity_kwh,
                charger_type=charger_type,
                charging_start_time=datetime.combine(charge_date, charge_time).isoformat(),
                charging_duration_hours=charging_duration_hours,
                charging_rate_kw=charging_rate_kw,
                charging_cost_usd=charging_cost_usd,
                soc_start_pct=soc_start_pct,
                soc_end_pct=soc_end_pct,
                distance_since_last_charge_km=distance_since_last_charge_km,
                temperature_c=temperature_c,
                vehicle_age_years=vehicle_age_years,
                user_type=user_type,
            )

            try:
                bundle = get_model_bundle()
                prediction = predict_session(session, bundle)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                physics_estimate = battery_capacity_kwh * (soc_end_pct - soc_start_pct) / 100 / 0.9

                col_a, col_b = st.columns(2)
                col_a.metric("Predicted energy consumed", f"{prediction:.2f} kWh")
                col_b.metric("Physics estimate (battery x delta SoC / efficiency)", f"~{physics_estimate:.2f} kWh")
                st.caption(f"Served by model: **{bundle['model_name']}**")
