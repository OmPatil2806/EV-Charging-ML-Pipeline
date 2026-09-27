"""Data access layer: cached loaders for config and the processed dataset.

Only Streamlit caching decorators are used here - no rendering (st.write,
st.error, st.warning, etc.) belongs in this layer. Pages call these
functions and decide how to present the result, including its absence.
"""

import pandas as pd
import streamlit as st

from src.utils.config import load_config, resolve_path


@st.cache_data
def get_config() -> dict:
    return load_config()


@st.cache_data
def load_processed_data() -> pd.DataFrame | None:
    cfg = get_config()
    path = resolve_path(cfg["data"]["processed_path"])
    if not path.exists():
        return None
    return pd.read_csv(path)


def filter_sessions(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filter selections to a dataframe. Empty/missing filter
    values mean "no restriction" for that column."""
    filtered = df
    if filters.get("vehicle_model"):
        filtered = filtered[filtered["vehicle_model"].isin(filters["vehicle_model"])]
    if filters.get("charger_type"):
        filtered = filtered[filtered["charger_type"].isin(filters["charger_type"])]
    return filtered
