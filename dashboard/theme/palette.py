"""Validated chart color tokens (dataviz skill reference palette, light mode).

Streamlit's own widget theme lives in .streamlit/config.toml; this file is
only for chart marks (plotly), which config.toml doesn't reach. Categorical
order is fixed and validated for adjacent-pair colorblind safety - don't
reorder it.
"""

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DIVERGING_POSITIVE = "#2a78d6"
DIVERGING_NEGATIVE = "#e34948"
SEQUENTIAL = "#2a78d6"
SURFACE = "#fcfcfb"
