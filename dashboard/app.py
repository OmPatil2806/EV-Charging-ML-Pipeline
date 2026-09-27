"""EV Charging Energy Prediction dashboard - entrypoint.

Layered as: pages/ (what the user sees) -> components/ (reusable render
functions) -> services/ (cached data/model access - the only layer that
touches disk or the model bundle) -> src/ (the pipeline/inference code
this dashboard is a UI shell over).

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.components.sidebar import render_sidebar  # noqa: E402

st.set_page_config(page_title="EV Charging Energy Prediction", layout="wide")

pages = [
    st.Page("pages/overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/data_exploration.py", title="Data Exploration", icon="📊"),
    st.Page("pages/model_comparison.py", title="Model Comparison", icon="🏆"),
    st.Page("pages/live_prediction.py", title="Live Prediction", icon="🔮"),
]

render_sidebar()
navigation = st.navigation(pages)
navigation.run()
