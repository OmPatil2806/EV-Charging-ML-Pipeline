"""Train all models (classical ML + neural net) on the feature-engineered
dataset, using a preprocessing pipeline fit only on the training split.
"""

from src.features.feature_engineering import build_feature_dataset
from src.models.neural_net import train_neural_net
from src.models.preprocessing import build_preprocessor, fit_transform_splits, split_data
from src.models.sklearn_models import train_sklearn_models
from src.utils.config import load_config


def train_all_models(cfg: dict = None) -> dict:
    cfg = cfg or load_config()

    df = build_feature_dataset(cfg)
    splits = split_data(df, cfg)

    preprocessor = build_preprocessor(cfg)
    X_train, X_val, X_test, preprocessor = fit_transform_splits(splits, preprocessor)

    sklearn_models = train_sklearn_models(X_train, splits["y_train"], cfg)

    nn_model, nn_history = train_neural_net(
        X_train,
        splits["y_train"].to_numpy(),
        X_val,
        splits["y_val"].to_numpy(),
        cfg,
    )

    return {
        "preprocessor": preprocessor,
        "sklearn_models": sklearn_models,
        "neural_net": nn_model,
        "neural_net_history": nn_history,
        "splits": splits,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
    }


def main():
    cfg = load_config()
    results = train_all_models(cfg)

    print(f"\nTrained {len(results['sklearn_models'])} sklearn models + 1 neural net.")
    print(
        f"Train rows: {results['X_train'].shape[0]}, "
        f"Val rows: {results['X_val'].shape[0]}, "
        f"Test rows: {results['X_test'].shape[0]}, "
        f"Features after preprocessing: {results['X_train'].shape[1]}"
    )


if __name__ == "__main__":
    main()
