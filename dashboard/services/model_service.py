"""Model access layer: cached loading of the trained model bundle/metrics,
and a thin wrapper around src.inference for scoring.

Only Streamlit caching decorators are used here - no rendering. Pages call
these functions and decide how to present the result, including its
absence.
"""

import json
from pathlib import Path

import streamlit as st

from dashboard.services.data_service import get_config
from src.inference import load_model_bundle, predict_session
from src.utils.config import resolve_path


@st.cache_resource
def get_model_bundle() -> dict:
    """Cached for the life of the server process - the Keras/sklearn model
    and preprocessor are loaded from disk exactly once, not once per page."""
    return load_model_bundle(get_config())


@st.cache_data
def get_metrics() -> dict | None:
    cfg = get_config()
    path = resolve_path(cfg["artifacts"]["dir"]) / cfg["artifacts"]["metrics_file"]
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def artifacts_available() -> bool:
    return get_metrics() is not None


def current_model_name() -> str | None:
    metrics = get_metrics()
    return metrics["best_model"] if metrics else None


def get_residual_plot_path() -> Path | None:
    cfg = get_config()
    path = resolve_path(cfg["artifacts"]["dir"]) / cfg["artifacts"]["residual_plot_file"]
    return path if path.exists() else None


def score_session(session: dict) -> float:
    bundle = get_model_bundle()
    return predict_session(session, bundle)
