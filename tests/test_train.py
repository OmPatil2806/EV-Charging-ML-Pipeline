import numpy as np
import pandas as pd

from src.models.train import train_all_models

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
            "epochs": 3,
            "batch_size": 8,
            "early_stopping_patience": 2,
        },
    },
}


def _sample_df(n=120):
    rng = np.random.default_rng(0)
    num1 = rng.normal(size=n)
    num2 = rng.normal(size=n)
    cat1 = rng.choice(["a", "b", "c"], size=n)
    # a target with genuine signal, like the real pipeline's physics feature
    target = 3 * num1 - 2 * num2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"num1": num1, "num2": num2, "cat1": cat1, "target": target})


def test_train_all_models_runs_end_to_end_and_returns_expected_structure():
    df = _sample_df()

    results = train_all_models(CFG, df=df)

    assert set(results["sklearn_models"].keys()) == {
        "linear_regression",
        "ridge",
        "random_forest",
        "gradient_boosting",
    }
    assert results["neural_net"] is not None
    assert results["X_train"].shape[0] + results["X_val"].shape[0] + results["X_test"].shape[0] == len(df)


def test_train_all_models_produces_usable_predictions_on_test_set():
    df = _sample_df()

    results = train_all_models(CFG, df=df)
    X_test = results["X_test"]

    for model in results["sklearn_models"].values():
        preds = model.predict(X_test)
        assert len(preds) == X_test.shape[0]

    nn_preds = results["neural_net"].predict(X_test, verbose=0)
    assert nn_preds.shape[0] == X_test.shape[0]
