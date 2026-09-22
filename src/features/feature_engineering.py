"""Feature construction: time features, SoC change, and the physics-based feature.

Categorical encoding and scaling are deliberately NOT done here - they're
fit inside the training pipeline, on the training split only, so no
test-set statistics leak into the transform. This module only builds
deterministic, per-row features that don't depend on the rest of the
dataset.
"""

import pandas as pd

from src.data.cleaning import clean_data
from src.data.ingestion import load_raw_data
from src.utils.config import load_config, resolve_path
from src.utils.time_features import WEEKEND_DAYS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start_time = pd.to_datetime(df["charging_start_time"])
    df["start_hour"] = start_time.dt.hour
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND_DAYS).astype(int)
    return df


def add_soc_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soc_change_pct"] = df["soc_end_pct"] - df["soc_start_pct"]
    return df


def add_physics_feature(df: pd.DataFrame) -> pd.DataFrame:
    """expected_energy_kwh: energy physically delivered to the battery
    (Battery Capacity x SoC change / 100) - the strongest expected predictor
    of Energy Consumed, since that's the actual physical relationship.
    """
    df = df.copy()
    if "soc_change_pct" not in df.columns:
        df = add_soc_change(df)
    df["expected_energy_kwh"] = df["battery_capacity_kwh"] * df["soc_change_pct"] / 100
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_soc_change(df)
    df = add_physics_feature(df)
    df = df.drop(columns=["session_id", "user_id", "charging_start_time"])
    return df


def build_feature_dataset(cfg: dict = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = load_raw_data(cfg)
    df = clean_data(df, cfg)
    df = engineer_features(df)
    return df


def main():
    cfg = load_config()
    df = build_feature_dataset(cfg)

    processed_path = resolve_path(cfg["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"Feature-engineered dataset: {df.shape[0]} rows, {df.shape[1]} columns -> {processed_path}")
    print(f"Columns: {list(df.columns)}")

    target = cfg["target_column"]
    correlations = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    print("\nCorrelation with target:")
    print(correlations)


if __name__ == "__main__":
    main()
