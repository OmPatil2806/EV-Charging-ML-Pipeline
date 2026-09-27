"""Shared sidebar: model/artifact status + global data filters.

Rendered once by app.py so it's consistent across every page. Filter
selections are written to session state (dashboard.state) so every page
reads the same filter values without re-deriving them.
"""

import streamlit as st

from dashboard.services import data_service, model_service
from dashboard.state import get_filters, set_filters


def render_sidebar() -> None:
    st.sidebar.title("EV Charging ML")

    metrics = model_service.get_metrics()
    if metrics:
        st.sidebar.success(f"Model loaded: **{metrics['best_model']}**")
    else:
        st.sidebar.warning("No trained model found.")
        st.sidebar.code("python pipeline/run_pipeline.py", language="bash")

    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    df = data_service.load_processed_data()
    filters = get_filters()

    if df is not None:
        vehicle_models = sorted(df["vehicle_model"].unique())
        charger_types = sorted(df["charger_type"].unique())

        selected_vehicles = st.sidebar.multiselect(
            "Vehicle model", vehicle_models, default=filters.get("vehicle_model", [])
        )
        selected_chargers = st.sidebar.multiselect(
            "Charger type", charger_types, default=filters.get("charger_type", [])
        )
        set_filters({"vehicle_model": selected_vehicles, "charger_type": selected_chargers})
    else:
        st.sidebar.caption("Filters appear once data is generated.")
