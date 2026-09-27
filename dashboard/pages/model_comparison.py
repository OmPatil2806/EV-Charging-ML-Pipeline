"""Model Comparison page: test-set leaderboard, RMSE chart, residual plots."""

import pandas as pd
import streamlit as st

from dashboard.components.charts import render_leaderboard_bar
from dashboard.services import model_service

st.title("Model Comparison")

metrics = model_service.get_metrics()
if metrics is None:
    st.warning("No trained models found. Run `python pipeline/run_pipeline.py` first.")
    st.stop()

leaderboard = metrics["leaderboard"]
best_name = metrics["best_model"]

board_df = pd.DataFrame(
    [{"model": name, **values} for name, values in leaderboard.items()]
).sort_values("rmse")

st.subheader("Test set leaderboard")
st.dataframe(
    board_df.rename(columns={"model": "Model", "rmse": "RMSE (kWh)", "mae": "MAE (kWh)", "r2": "R2"}),
    width="stretch",
    hide_index=True,
)

st.subheader("RMSE by model (lower is better)")
render_leaderboard_bar(board_df)

st.success(f"Best model (lowest test RMSE): **{best_name}**")

residual_path = model_service.get_residual_plot_path()
if residual_path:
    st.subheader("Residuals (actual - predicted)")
    st.image(str(residual_path), width="stretch")
