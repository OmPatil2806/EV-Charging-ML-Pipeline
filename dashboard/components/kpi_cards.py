"""Reusable KPI metric-card row."""

import streamlit as st


def render_kpi_row(cards: list[tuple[str, str]]) -> None:
    """cards: list of (label, value) pairs, rendered as equal-width columns."""
    columns = st.columns(len(cards))
    for col, (label, value) in zip(columns, cards):
        col.metric(label, value)
