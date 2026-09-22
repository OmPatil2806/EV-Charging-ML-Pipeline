"""Data cleaning: duplicates, missing values, and out-of-range readings.

Missing values in the target are dropped rather than imputed (an imputed
label is fabricated ground truth and would bias training); missing values
in feature columns are median-imputed, which is safe since the target is
never touched by it.
"""

import pandas as pd

from src.data.ingestion import load_raw_data
from src.utils.config import load_config


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows")
    return df


def drop_missing_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Dropped {removed} rows with missing target ('{target_column}')")
    return df


def impute_missing_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            continue
        n_missing = df[column].isnull().sum()
        if n_missing == 0:
            continue
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)
        print(f"Filled {n_missing} missing values in '{column}' with median {median_value:.2f}")
    return df


def cap_percentage_range(df: pd.DataFrame, columns: list, low: float = 0, high: float = 100) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        n_invalid = ((df[column] < low) | (df[column] > high)).sum()
        if n_invalid:
            df[column] = df[column].clip(lower=low, upper=high)
            print(f"Capped {n_invalid} out-of-range values in '{column}' to [{low}, {high}]")
    return df


def clip_non_negative(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        n_negative = (df[column] < 0).sum()
        if n_negative:
            df[column] = df[column].clip(lower=0)
            print(f"Clipped {n_negative} negative values in '{column}' to 0")
    return df


def clean_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    target_column = cfg["target_column"]
    numeric_feature_columns = [c for c in cfg["numeric_features"] if c in df.columns]

    df = drop_duplicate_rows(df)
    df = drop_missing_target(df, target_column)
    df = impute_missing_numeric(df, numeric_feature_columns)
    df = cap_percentage_range(df, ["soc_start_pct", "soc_end_pct"])
    df = clip_non_negative(df, ["charging_duration_hours", "charging_cost_usd", target_column])

    return df


def main():
    cfg = load_config()
    df = load_raw_data(cfg)
    print(f"Raw rows: {len(df)}")

    df_clean = clean_data(df, cfg)
    print(f"Clean rows: {len(df_clean)}")
    print(f"Remaining nulls: {int(df_clean.isnull().sum().sum())}")


if __name__ == "__main__":
    main()
