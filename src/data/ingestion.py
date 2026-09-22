"""Load the raw EV charging sessions CSV and report basic schema/quality stats."""

import pandas as pd

from src.utils.config import load_config, resolve_path


def load_raw_data(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw_path = resolve_path(cfg["data"]["raw_path"])
    return pd.read_csv(raw_path, parse_dates=["charging_start_time"])


def summarize(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def main():
    df = load_raw_data()
    summary = summarize(df)

    print(f"Rows: {summary['n_rows']}, Columns: {summary['n_columns']}")
    print("Null counts (non-zero only):")
    for column, count in summary["null_counts"].items():
        if count:
            print(f"  {column}: {count}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")


if __name__ == "__main__":
    main()
