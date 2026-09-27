"""The charging-session input form.

Renders the form and, on submit, returns the collected session dict - it
does not call the model itself (that's the page's job, via
dashboard.services.model_service). Keeps this component testable/reusable
independent of which model is currently loaded.
"""

from datetime import datetime

import streamlit as st

VEHICLE_MODELS = ["Tesla Model 3", "Nissan Leaf", "Chevy Bolt", "BMW i3", "Hyundai Kona"]
CHARGER_TYPES = ["Level 1", "Level 2", "DC Fast Charger"]
USER_TYPES = ["Commuter", "Casual Driver", "Long-Distance Traveler"]


def render_prediction_form() -> dict | None:
    """Returns the session dict if the form was submitted this run, else None."""
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            vehicle_model = st.selectbox("Vehicle model", VEHICLE_MODELS)
            battery_capacity_kwh = st.number_input("Battery capacity (kWh)", min_value=1.0, value=60.0)
            charger_type = st.selectbox("Charger type", CHARGER_TYPES, index=1)  # Level 2
            user_type = st.selectbox("User type", USER_TYPES)
        with c2:
            charge_date = st.date_input("Charging date", value=datetime.now().date())
            charge_time = st.time_input("Charging time", value=datetime.now().time())
            # defaults are internally consistent: 60 kWh x 60% SoC gain / ~0.9
            # efficiency = ~40 kWh, at an 11 kW (Level 2) rate = ~3.6 hours
            charging_duration_hours = st.number_input("Charging duration (hours)", min_value=0.01, value=3.6)
            charging_rate_kw = st.number_input("Charging rate (kW)", min_value=0.1, value=11.0)
        with c3:
            charging_cost_usd = st.number_input("Charging cost (USD)", min_value=0.0, value=9.0)
            soc_start_pct = st.slider("SoC start (%)", 0, 100, 20)
            soc_end_pct = st.slider("SoC end (%)", 0, 100, 80)
            distance_since_last_charge_km = st.number_input(
                "Distance since last charge (km)", min_value=0.0, value=150.0
            )

        col_age, col_temp = st.columns(2)
        with col_age:
            vehicle_age_years = st.number_input("Vehicle age (years)", min_value=0.0, value=2.0)
        with col_temp:
            temperature_c = st.number_input("Temperature (C)", value=20.0)

        submitted = st.form_submit_button("Predict energy consumed")

    if not submitted:
        return None

    return dict(
        vehicle_model=vehicle_model,
        battery_capacity_kwh=battery_capacity_kwh,
        charger_type=charger_type,
        charging_start_time=datetime.combine(charge_date, charge_time).isoformat(),
        charging_duration_hours=charging_duration_hours,
        charging_rate_kw=charging_rate_kw,
        charging_cost_usd=charging_cost_usd,
        soc_start_pct=soc_start_pct,
        soc_end_pct=soc_end_pct,
        distance_since_last_charge_km=distance_since_last_charge_km,
        temperature_c=temperature_c,
        vehicle_age_years=vehicle_age_years,
        user_type=user_type,
    )
