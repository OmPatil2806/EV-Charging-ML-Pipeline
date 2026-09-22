"""Evaluate all trained models on the held-out test set: leaderboard,
feature importances, residual plots, and persist the best model +
preprocessing artifact for serving.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.train import train_all_models
from src.utils.config import load_config, resolve_path


def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_models(results: dict):
    X_test = results["X_test"]
    y_test = results["splits"]["y_test"].to_numpy()

    metrics = {}
    predictions = {}

    for name, model in results["sklearn_models"].items():
        preds = model.predict(X_test)
        predictions[name] = preds
        metrics[name] = compute_metrics(y_test, preds)

    nn_preds = results["neural_net"].predict(X_test, verbose=0).ravel()
    predictions["neural_net"] = nn_preds
    metrics["neural_net"] = compute_metrics(y_test, nn_preds)

    return metrics, predictions, y_test


def print_leaderboard(metrics: dict):
    print(f"{'Model':<20}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    for name, m in sorted(metrics.items(), key=lambda kv: kv[1]["rmse"]):
        print(f"{name:<20}{m['rmse']:>10.3f}{m['mae']:>10.3f}{m['r2']:>10.4f}")


def get_feature_importances(results: dict) -> dict:
    preprocessor = results["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    importances = {}
    for name in ["random_forest", "gradient_boosting"]:
        model = results["sklearn_models"][name]
        importances[name] = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
    return importances


def plot_residuals(predictions: dict, y_test, out_path):
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, (name, preds) in zip(axes, predictions.items()):
        residuals = y_test - preds
        ax.scatter(preds, residuals, alpha=0.4, s=10)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Predicted (kWh)")
    axes[0].set_ylabel("Residual (actual - predicted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def select_best_model(results: dict, metrics: dict):
    best_name = min(metrics, key=lambda name: metrics[name]["rmse"])
    if best_name == "neural_net":
        best_model = results["neural_net"]
    else:
        best_model = results["sklearn_models"][best_name]
    return best_name, best_model


def save_artifacts(best_name, best_model, preprocessor, metrics, cfg):
    artifacts_dir = resolve_path(cfg["artifacts"]["dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, artifacts_dir / cfg["artifacts"]["preprocessor_file"])

    if best_name == "neural_net":
        model_path = artifacts_dir / f"{cfg['artifacts']['best_model_file']}.keras"
        best_model.save(model_path)
    else:
        model_path = artifacts_dir / f"{cfg['artifacts']['best_model_file']}.pkl"
        joblib.dump(best_model, model_path)

    metrics_out = {"leaderboard": metrics, "best_model": best_name}
    with open(artifacts_dir / cfg["artifacts"]["metrics_file"], "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nSaved best model ('{best_name}') -> {model_path}")
    print(f"Saved preprocessor -> {artifacts_dir / cfg['artifacts']['preprocessor_file']}")
    print(f"Saved metrics -> {artifacts_dir / cfg['artifacts']['metrics_file']}")


def main():
    cfg = load_config()
    results = train_all_models(cfg)

    metrics, predictions, y_test = evaluate_models(results)
    print("\n=== Test Set Leaderboard ===")
    print_leaderboard(metrics)

    importances = get_feature_importances(results)
    print("\n=== Top 5 Feature Importances ===")
    for name, ranked in importances.items():
        print(f"\n{name}:")
        for feature, importance in ranked[:5]:
            print(f"  {feature}: {importance:.4f}")

    artifacts_dir = resolve_path(cfg["artifacts"]["dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_residuals(predictions, y_test, artifacts_dir / cfg["artifacts"]["residual_plot_file"])

    best_name, best_model = select_best_model(results, metrics)
    save_artifacts(best_name, best_model, results["preprocessor"], metrics, cfg)


if __name__ == "__main__":
    main()
