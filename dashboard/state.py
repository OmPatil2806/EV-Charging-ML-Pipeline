"""Typed accessors over st.session_state, so pages/components don't poke at
raw string keys directly.
"""

import streamlit as st

_FILTERS_KEY = "sidebar_filters"
_PREDICTION_HISTORY_KEY = "prediction_history"
_MAX_HISTORY = 20


def get_filters() -> dict:
    return st.session_state.get(_FILTERS_KEY, {})


def set_filters(filters: dict) -> None:
    st.session_state[_FILTERS_KEY] = filters


def get_prediction_history() -> list:
    return st.session_state.setdefault(_PREDICTION_HISTORY_KEY, [])


def add_prediction(record: dict) -> None:
    history = get_prediction_history()
    history.insert(0, record)
    st.session_state[_PREDICTION_HISTORY_KEY] = history[:_MAX_HISTORY]
