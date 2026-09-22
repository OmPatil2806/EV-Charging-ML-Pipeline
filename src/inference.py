"""Shared inference logic: load the trained model bundle and score a single
session. Used by both the FastAPI service (api/main.py) and the Streamlit
dashboard so there's one implementation of "how to score a session," not two
that could drift apart.
"""

import json

import joblib
import pandas as pd

from src.features.feature_engineering import add_physics_feature, add_time_features
from src.utils.config import load_config, resolve_path
from src.utils.time_features import derive_time_fields


def load_model_bundle(cfg: dict = None) -> dict:
    cfg = cfg or load_config()
    artifacts_dir = resolve_path(cfg["artifacts"]["dir"])

    preprocessor_path = artifacts_dir / cfg["artifacts"]["preprocessor_file"]
    metrics_path = artifacts_dir / cfg["artifacts"]["metrics_file"]

    if not preprocessor_path.exists() or not metrics_path.exists():
        raise RuntimeError(
            f"No trained artifacts found in {artifacts_dir}. "
            "Run `python pipeline/run_pipeline.py` first."
        )

    with open(metrics_path) as f:
        metrics = json.load(f)
    best_name = metrics["best_model"]

    if best_name == "neural_net":
        from tensorflow import keras

        model = keras.models.load_model(artifacts_dir / f"{cfg['artifacts']['best_model_file']}.keras")
    else:
        model = joblib.load(artifacts_dir / f"{cfg['artifacts']['best_model_file']}.pkl")

    return {
        "cfg": cfg,
        "preprocessor": joblib.load(preprocessor_path),
        "model": model,
        "model_name": best_name,
        "leaderboard": metrics["leaderboard"],
    }


def predict_session(session: dict, bundle: dict) -> float:
    cfg = bundle["cfg"]
    row = dict(session)

    timestamp = pd.Timestamp(row["charging_start_time"])
    row.update(derive_time_fields(timestamp))

    df = pd.DataFrame([row])
    df = add_time_features(df)
    df = add_physics_feature(df)

    feature_columns = cfg["numeric_features"] + cfg["categorical_features"]
    X = bundle["preprocessor"].transform(df[feature_columns])

    if bundle["model_name"] == "neural_net":
        prediction = float(bundle["model"].predict(X, verbose=0).ravel()[0])
    else:
        prediction = float(bundle["model"].predict(X)[0])

    return max(prediction, 0.0)
