"""Chart-rendering components.

Pure rendering only - no data loading here; callers (pages) pass in
already-loaded/filtered dataframes. Uses the validated categorical/
diverging/sequential color tokens from dashboard.theme.palette.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme.palette import (
    CATEGORICAL,
    DIVERGING_NEGATIVE,
    DIVERGING_POSITIVE,
    SEQUENTIAL,
    SURFACE,
)


def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, margin=dict(t=30, b=10))
    return fig


def render_histogram(df: pd.DataFrame, column: str) -> None:
    fig = px.histogram(df, x=column, nbins=40, color_discrete_sequence=[SEQUENTIAL])
    fig.update_layout(bargap=0.05)
    st.plotly_chart(_style(fig), width="stretch")


def render_correlation_bar(corr: pd.Series) -> None:
    colors = [DIVERGING_NEGATIVE if v < 0 else DIVERGING_POSITIVE for v in corr.values]
    fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color=colors))
    fig.update_layout(xaxis_title="Correlation", yaxis_title="")
    st.plotly_chart(_style(fig), width="stretch")


def render_category_counts(df: pd.DataFrame, column: str) -> None:
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", color=column, color_discrete_sequence=CATEGORICAL)
    fig.update_layout(showlegend=False)
    st.plotly_chart(_style(fig), width="stretch")


def render_leaderboard_bar(board_df: pd.DataFrame) -> None:
    fig = px.bar(board_df, x="model", y="rmse", color="model", color_discrete_sequence=CATEGORICAL)
    fig.update_layout(showlegend=False, yaxis_title="RMSE (kWh)", xaxis_title="")
    st.plotly_chart(_style(fig), width="stretch")
