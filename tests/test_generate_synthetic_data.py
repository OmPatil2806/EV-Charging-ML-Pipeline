import numpy as np

from data.generate_synthetic_data import generate_dataset, inject_data_quality_issues

EXPECTED_COLUMNS = {
    "session_id",
    "user_id",
    "vehicle_model",
    "battery_capacity_kwh",
    "charger_type",
    "charging_start_time",
    "time_of_day",
    "day_of_week",
    "charging_duration_hours",
    "charging_rate_kw",
    "charging_cost_usd",
    "soc_start_pct",
    "soc_end_pct",
    "distance_since_last_charge_km",
    "temperature_c",
    "vehicle_age_years",
    "user_type",
    "energy_consumed_kwh",
}


def test_generate_dataset_has_expected_columns_and_row_count():
    df = generate_dataset(n=200, seed=0)

    assert len(df) == 200
    assert set(df.columns) == EXPECTED_COLUMNS


def test_generate_dataset_values_are_physically_sane():
    df = generate_dataset(n=500, seed=0)

    assert (df["energy_consumed_kwh"] > 0).all()
    assert (df["charging_duration_hours"] > 0).all()
    assert (df["charging_rate_kw"] > 0).all()
    assert df["soc_start_pct"].between(0, 100).all()
    assert df["soc_end_pct"].between(0, 100).all()


def test_generate_dataset_has_genuine_physical_signal():
    df = generate_dataset(n=1000, seed=0)

    physics_estimate = df["battery_capacity_kwh"] * (df["soc_end_pct"] - df["soc_start_pct"])
    correlation = np.corrcoef(df["energy_consumed_kwh"], physics_estimate)[0, 1]

    # unlike a purely random target, this should correlate strongly
    assert correlation > 0.6


def test_generate_dataset_is_deterministic_given_a_seed():
    df1 = generate_dataset(n=100, seed=42)
    df2 = generate_dataset(n=100, seed=42)

    assert df1["energy_consumed_kwh"].tolist() == df2["energy_consumed_kwh"].tolist()


def test_inject_data_quality_issues_adds_nulls_outliers_and_duplicates():
    cfg = {"data": {"missing_value_rate": 0.5, "outlier_rate": 0.5}}
    df = generate_dataset(n=200, seed=0)
    rng = np.random.default_rng(1)

    dirty = inject_data_quality_issues(df, rng, cfg)

    assert len(dirty) > len(df)  # duplicates were appended
    assert dirty["energy_consumed_kwh"].isnull().sum() > 0
    assert dirty["charging_rate_kw"].isnull().sum() > 0
    assert dirty["distance_since_last_charge_km"].isnull().sum() > 0
    out_of_range = (dirty["soc_start_pct"] < 0) | (dirty["soc_start_pct"] > 100)
    assert out_of_range.sum() > 0
