"""Preprocessing pipeline and train/val/test split.

The ColumnTransformer here is fit ONLY on the training split (via
fit_transform_splits), so encoder categories and scaler statistics never
see validation or test data - this is the leakage fix flagged earlier
against the original notebook, where StringIndexer was fit on the full
dataset before splitting.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), cfg["numeric_features"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cfg["categorical_features"],
            ),
        ]
    )


def split_data(df: pd.DataFrame, cfg: dict) -> dict:
    target_column = cfg["target_column"]
    feature_columns = cfg["numeric_features"] + cfg["categorical_features"]

    X = df[feature_columns]
    y = df[target_column]

    test_size = cfg["train_test_split"]["test_size"]
    val_size = cfg["train_test_split"]["val_size"]
    seed = cfg["random_seed"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=relative_val_size, random_state=seed
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def fit_transform_splits(splits: dict, preprocessor: ColumnTransformer):
    X_train = preprocessor.fit_transform(splits["X_train"])
    X_val = preprocessor.transform(splits["X_val"])
    X_test = preprocessor.transform(splits["X_test"])
    return X_train, X_val, X_test, preprocessor
