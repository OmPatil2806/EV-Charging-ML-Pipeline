"""Overview page: at-a-glance KPIs and pipeline/model status."""

import streamlit as st

from dashboard.components.kpi_cards import render_kpi_row
from dashboard.services import data_service, model_service

st.title("Overview")

df = data_service.load_processed_data()
metrics = model_service.get_metrics()

if df is None or metrics is None:
    st.warning("Pipeline hasn't been run yet.")
    st.code("python pipeline/run_pipeline.py", language="bash")
    st.stop()

best_name = metrics["best_model"]
best_metrics = metrics["leaderboard"][best_name]

render_kpi_row(
    [
        ("Sessions", f"{len(df):,}"),
        ("Best model", best_name),
        ("Test R2", f"{best_metrics['r2']:.3f}"),
        ("Test RMSE", f"{best_metrics['rmse']:.2f} kWh"),
    ]
)

st.divider()
st.subheader("What this project does")
st.write(
    "Predicts energy consumed (kWh) for an EV charging session from session characteristics, "
    "comparing 5 classical ML and deep learning models. Use the sidebar to navigate: explore "
    "the data, compare model performance, or try a live prediction."
)
