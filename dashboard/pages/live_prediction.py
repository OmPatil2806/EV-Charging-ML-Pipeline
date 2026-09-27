"""Live Prediction page: score a hypothetical session and keep a short
history of recent tries (this browser session only) for comparison.
"""

import streamlit as st

from dashboard.components.prediction_form import render_prediction_form
from dashboard.services import model_service
from dashboard.state import add_prediction, get_prediction_history

st.title("Live Prediction")

if not model_service.artifacts_available():
    st.warning("No trained models found. Run `python pipeline/run_pipeline.py` first.")
    st.stop()

st.write("Enter a charging session's characteristics to predict energy consumed.")

session = render_prediction_form()

if session is not None:
    try:
        prediction = model_service.score_session(session)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        physics_estimate = (
            session["battery_capacity_kwh"]
            * (session["soc_end_pct"] - session["soc_start_pct"])
            / 100
            / 0.9
        )
        model_name = model_service.current_model_name()

        col_a, col_b = st.columns(2)
        col_a.metric("Predicted energy consumed", f"{prediction:.2f} kWh")
        col_b.metric("Physics estimate (battery x delta SoC / efficiency)", f"~{physics_estimate:.2f} kWh")
        st.caption(f"Served by model: **{model_name}**")

        add_prediction(
            {
                "vehicle_model": session["vehicle_model"],
                "charger_type": session["charger_type"],
                "battery_capacity_kwh": session["battery_capacity_kwh"],
                "soc_change_pct": session["soc_end_pct"] - session["soc_start_pct"],
                "predicted_kwh": round(prediction, 2),
                "model": model_name,
            }
        )

history = get_prediction_history()
if history:
    st.divider()
    st.subheader("Recent predictions (this session)")
    st.dataframe(history, width="stretch", hide_index=True)
