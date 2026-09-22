import numpy as np
import pandas as pd

from src.models.preprocessing import build_preprocessor, split_data
from src.models.sklearn_models import get_model_grid, train_sklearn_models

CFG = {
    "target_column": "target",
    "numeric_features": ["num1", "num2"],
    "categorical_features": ["cat1"],
    "train_test_split": {"test_size": 0.2, "val_size": 0.2},
    "random_seed": 42,
    "models": {
        "ridge": {"alpha": [1.0]},
        "random_forest": {"n_estimators": [10], "max_depth": [4]},
        "gradient_boosting": {"n_estimators": [10], "learning_rate": [0.1], "max_depth": [2]},
        "neural_net": {
            "hidden_layers": [4],
            "dropout": 0.1,
            "learning_rate": 0.01,
            "epochs": 2,
            "batch_size": 8,
            "early_stopping_patience": 2,
        },
    },
}


def _sample_df(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(["a", "b", "c"], size=n),
            "target": rng.normal(size=n),
        }
    )


def test_split_data_respects_proportions_and_no_overlap():
    df = _sample_df(100)
    splits = split_data(df, CFG)

    assert len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]) == 100
    train_idx = set(splits["X_train"].index)
    val_idx = set(splits["X_val"].index)
    test_idx = set(splits["X_test"].index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_build_preprocessor_encodes_and_scales():
    df = _sample_df(50)
    preprocessor = build_preprocessor(CFG)
    transformed = preprocessor.fit_transform(df[["num1", "num2", "cat1"]])

    # 2 scaled numeric columns + 3 one-hot categorical columns (a, b, c)
    assert transformed.shape == (50, 5)


def test_preprocessor_fit_on_train_only_ignores_unseen_test_categories():
    train_df = pd.DataFrame({"num1": [1.0, 2.0], "num2": [3.0, 4.0], "cat1": ["a", "b"]})
    test_df = pd.DataFrame({"num1": [5.0], "num2": [6.0], "cat1": ["unseen_category"]})

    preprocessor = build_preprocessor(CFG)
    preprocessor.fit(train_df)
    transformed = preprocessor.transform(test_df)

    # unseen category is encoded as all-zeros (handle_unknown="ignore"), not an error
    # (train only saw categories "a" and "b" -> 2 numeric + 2 one-hot columns)
    assert transformed.shape == (1, 4)


def test_get_model_grid_returns_expected_models():
    grid = get_model_grid(CFG)
    assert set(grid.keys()) == {"linear_regression", "ridge", "random_forest", "gradient_boosting"}


def test_train_sklearn_models_fits_all_models_on_tiny_data():
    df = _sample_df(60)
    preprocessor = build_preprocessor(CFG)
    X = preprocessor.fit_transform(df[["num1", "num2", "cat1"]])
    y = df["target"]

    trained = train_sklearn_models(X, y, CFG, cv=2)

    assert set(trained.keys()) == {"linear_regression", "ridge", "random_forest", "gradient_boosting"}
    for model in trained.values():
        preds = model.predict(X)
        assert len(preds) == len(y)


def test_build_neural_net_has_expected_input_and_output_shape():
    from src.models.neural_net import build_neural_net

    model = build_neural_net(input_dim=5, cfg=CFG)

    assert model.input_shape == (None, 5)
    assert model.output_shape == (None, 1)


def test_train_neural_net_runs_and_returns_history():
    from src.models.neural_net import train_neural_net

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 4)).astype("float32")
    y_train = rng.normal(size=20).astype("float32")
    X_val = rng.normal(size=(6, 4)).astype("float32")
    y_val = rng.normal(size=6).astype("float32")

    model, history = train_neural_net(X_train, y_train, X_val, y_val, CFG)

    assert "loss" in history.history
    preds = model.predict(X_val, verbose=0)
    assert preds.shape == (6, 1)
