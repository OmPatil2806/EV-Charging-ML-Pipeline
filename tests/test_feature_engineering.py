import pandas as pd

from src.features.feature_engineering import (
    add_physics_feature,
    add_soc_change,
    add_time_features,
    engineer_features,
)


def _sample_df():
    return pd.DataFrame(
        {
            "session_id": ["S1", "S2"],
            "user_id": ["U1", "U2"],
            "charging_start_time": ["2024-01-06 14:30:00", "2024-01-08 09:00:00"],  # Sat, Mon
            "day_of_week": ["Saturday", "Monday"],
            "battery_capacity_kwh": [60.0, 40.0],
            "soc_start_pct": [20.0, 30.0],
            "soc_end_pct": [70.0, 50.0],
        }
    )


def test_add_time_features_extracts_hour_and_weekend_flag():
    df = add_time_features(_sample_df())
    assert df["start_hour"].tolist() == [14, 9]
    assert df["is_weekend"].tolist() == [1, 0]


def test_add_soc_change_computes_difference():
    df = add_soc_change(_sample_df())
    assert df["soc_change_pct"].tolist() == [50.0, 20.0]


def test_add_physics_feature_matches_expected_formula():
    df = add_physics_feature(_sample_df())
    assert df["expected_energy_kwh"].tolist() == [60.0 * 50.0 / 100, 40.0 * 20.0 / 100]


def test_add_physics_feature_works_without_precomputed_soc_change():
    df = _sample_df().drop(columns=[])  # soc_change_pct not present yet
    result = add_physics_feature(df)
    assert "soc_change_pct" in result.columns
    assert "expected_energy_kwh" in result.columns


def test_engineer_features_drops_identifiers_and_adds_engineered_columns():
    df = engineer_features(_sample_df())
    for column in ["session_id", "user_id", "charging_start_time"]:
        assert column not in df.columns
    for column in ["start_hour", "is_weekend", "soc_change_pct", "expected_energy_kwh"]:
        assert column in df.columns
