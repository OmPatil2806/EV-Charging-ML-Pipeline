"""Data Exploration page: distributions, correlation with target, category
counts. Respects the global sidebar filters (vehicle model, charger type).
"""

import streamlit as st

from dashboard.components.charts import render_category_counts, render_correlation_bar, render_histogram
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.services.data_service import filter_sessions, get_config, load_processed_data
from dashboard.state import get_filters

st.title("Data Exploration")

df = load_processed_data()
if df is None:
    st.warning("No processed dataset found. Run `python pipeline/run_pipeline.py` first.")
    st.stop()

cfg = get_config()
target = cfg["target_column"]

filtered = filter_sessions(df, get_filters())
if filtered.empty:
    st.info("No sessions match the current sidebar filters.")
    st.stop()

render_kpi_row(
    [
        ("Sessions (filtered)", f"{len(filtered):,} / {len(df):,}"),
        ("Features", str(df.shape[1] - 1)),
        ("Avg. Energy Consumed", f"{filtered[target].mean():.1f} kWh"),
    ]
)

st.subheader("Feature distribution")
numeric_cols = [c for c in cfg["numeric_features"] if c in filtered.columns]
default_idx = numeric_cols.index("expected_energy_kwh") if "expected_energy_kwh" in numeric_cols else 0
selected = st.selectbox("Numeric feature", numeric_cols, index=default_idx)
render_histogram(filtered, selected)

st.subheader("Correlation with Energy Consumed")
corr = filtered.corr(numeric_only=True)[target].drop(target).sort_values()
render_correlation_bar(corr)

st.subheader("Sessions by category")
cat_cols = [c for c in cfg["categorical_features"] if c in filtered.columns]
cat_selected = st.selectbox("Categorical feature", cat_cols)
render_category_counts(filtered, cat_selected)
