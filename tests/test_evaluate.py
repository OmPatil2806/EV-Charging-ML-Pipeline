import numpy as np

from src.evaluation.evaluate import compute_metrics, select_best_model


def test_compute_metrics_matches_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_compute_metrics_penalizes_errors():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["rmse"] > 0
    assert metrics["mae"] == 2 / 3
    assert metrics["r2"] < 1.0


def test_select_best_model_picks_lowest_rmse_sklearn():
    results = {
        "sklearn_models": {"a": "model_a", "b": "model_b"},
        "neural_net": "model_nn",
    }
    metrics = {
        "a": {"rmse": 2.0},
        "b": {"rmse": 1.0},
        "neural_net": {"rmse": 3.0},
    }

    best_name, best_model = select_best_model(results, metrics)

    assert best_name == "b"
    assert best_model == "model_b"


def test_select_best_model_picks_neural_net_when_it_wins():
    results = {
        "sklearn_models": {"a": "model_a"},
        "neural_net": "model_nn",
    }
    metrics = {
        "a": {"rmse": 5.0},
        "neural_net": {"rmse": 0.5},
    }

    best_name, best_model = select_best_model(results, metrics)

    assert best_name == "neural_net"
    assert best_model == "model_nn"
