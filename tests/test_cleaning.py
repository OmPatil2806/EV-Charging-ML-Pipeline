import numpy as np
import pandas as pd

from src.data.cleaning import (
    cap_percentage_range,
    clean_data,
    clip_non_negative,
    drop_duplicate_rows,
    drop_missing_target,
    impute_missing_numeric,
)


def test_drop_duplicate_rows_removes_exact_duplicates():
    df = pd.DataFrame({"a": [1, 2, 2], "b": [10, 20, 20]})
    result = drop_duplicate_rows(df)
    assert len(result) == 2


def test_drop_missing_target_removes_only_null_target_rows():
    df = pd.DataFrame({"target": [1.0, np.nan, 3.0], "feature": [1, 2, 3]})
    result = drop_missing_target(df, "target")
    assert len(result) == 2
    assert result["target"].isnull().sum() == 0


def test_impute_missing_numeric_fills_with_median():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    result = impute_missing_numeric(df, ["x"])
    assert result["x"].isnull().sum() == 0
    assert result.loc[1, "x"] == 2.0  # median of [1, 3]


def test_impute_missing_numeric_ignores_missing_columns():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    result = impute_missing_numeric(df, ["not_in_df"])
    pd.testing.assert_frame_equal(result, df)


def test_cap_percentage_range_clips_out_of_range_values():
    df = pd.DataFrame({"soc_start_pct": [-10.0, 50.0, 150.0]})
    result = cap_percentage_range(df, ["soc_start_pct"])
    assert result["soc_start_pct"].tolist() == [0.0, 50.0, 100.0]


def test_clip_non_negative_clips_negatives_to_zero():
    df = pd.DataFrame({"cost": [-5.0, 0.0, 10.0]})
    result = clip_non_negative(df, ["cost"])
    assert result["cost"].tolist() == [0.0, 0.0, 10.0]


def test_clean_data_end_to_end():
    df = pd.DataFrame(
        {
            "energy_consumed_kwh": [10.0, 10.0, np.nan, 20.0],
            "battery_capacity_kwh": [60.0, 60.0, 60.0, np.nan],
            "soc_start_pct": [10.0, 10.0, 20.0, 150.0],
            "soc_end_pct": [50.0, 50.0, 60.0, 70.0],
            "charging_duration_hours": [1.0, 1.0, 2.0, 3.0],
            "charging_cost_usd": [5.0, 5.0, 6.0, 7.0],
        }
    )
    cfg = {
        "target_column": "energy_consumed_kwh",
        "numeric_features": ["battery_capacity_kwh"],
    }

    result = clean_data(df, cfg)

    # exact-duplicate row (row 1) and missing-target row (row 2) both dropped
    assert len(result) == 2
    assert result["energy_consumed_kwh"].isnull().sum() == 0
    assert result["battery_capacity_kwh"].isnull().sum() == 0
    assert result["soc_start_pct"].between(0, 100).all()
    assert result["soc_end_pct"].between(0, 100).all()
