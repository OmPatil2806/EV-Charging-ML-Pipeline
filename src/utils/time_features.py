"""Time-derived field helpers, shared by synthetic data generation and the
inference API so both derive day_of_week/time_of_day from a timestamp the
same way.
"""

import pandas as pd

WEEKEND_DAYS = {"Saturday", "Sunday"}


def time_of_day_bucket(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 18:
        return "Afternoon"
    if 18 <= hour < 22:
        return "Evening"
    return "Night"


def derive_time_fields(timestamp: pd.Timestamp) -> dict:
    return {
        "day_of_week": timestamp.day_name(),
        "time_of_day": time_of_day_bucket(timestamp.hour),
    }
