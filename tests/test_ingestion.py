import numpy as np
import pandas as pd

from src.data.ingestion import summarize


def test_summarize_reports_shape_nulls_and_duplicates():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 1.0],
            "b": ["x", "y", "z", "x"],
        }
    )
    # last row is an exact duplicate of the first

    summary = summarize(df)

    assert summary["n_rows"] == 4
    assert summary["n_columns"] == 2
    assert summary["columns"] == ["a", "b"]
    assert summary["null_counts"]["a"] == 1
    assert summary["null_counts"]["b"] == 0
    assert summary["duplicate_rows"] == 1


def test_summarize_on_clean_data_reports_zero_issues():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    summary = summarize(df)

    assert summary["duplicate_rows"] == 0
    assert all(count == 0 for count in summary["null_counts"].values())
