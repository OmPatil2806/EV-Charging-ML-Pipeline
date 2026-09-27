"""Smoke tests for each dashboard page using Streamlit's headless AppTest
framework. These catch import errors and unhandled exceptions in the UI
layer that unit tests on src/ can't reach - e.g. a page importing a
function that was renamed in a service, or a component call with the
wrong argument shape. They pass whether or not pipeline artifacts exist,
since every page handles that absence with st.warning + st.stop().
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"

PAGES = [
    DASHBOARD_DIR / "pages" / "overview.py",
    DASHBOARD_DIR / "pages" / "data_exploration.py",
    DASHBOARD_DIR / "pages" / "model_comparison.py",
    DASHBOARD_DIR / "pages" / "live_prediction.py",
]


@pytest.mark.parametrize("page_path", PAGES, ids=lambda p: p.name)
def test_page_renders_without_exception(page_path):
    at = AppTest.from_file(str(page_path), default_timeout=30)
    at.run()

    assert not at.exception
