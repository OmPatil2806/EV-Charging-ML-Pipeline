"""FastAPI inference service: loads the best trained model + preprocessor
(saved by src/evaluation/evaluate.py) and serves energy consumption
predictions for a charging session.

Run with:
    uvicorn api.main:app --reload
"""

import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import PredictionResponse, SessionInput
from src.features.feature_engineering import add_physics_feature, add_time_features
from src.utils.config import load_config, resolve_path
from src.utils.time_features import derive_time_fields

_state: dict = {}


def load_artifacts():
    cfg = load_config()
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

        model_path = artifacts_dir / f"{cfg['artifacts']['best_model_file']}.keras"
        model = keras.models.load_model(model_path)
    else:
        model_path = artifacts_dir / f"{cfg['artifacts']['best_model_file']}.pkl"
        model = joblib.load(model_path)

    _state["cfg"] = cfg
    _state["preprocessor"] = joblib.load(preprocessor_path)
    _state["model"] = model
    _state["model_name"] = best_name
    _state["metrics"] = metrics["leaderboard"][best_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="EV Charging Energy Prediction API",
    description="Predicts energy consumed (kWh) for an EV charging session.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "EV Charging Energy Prediction API",
        "model": _state.get("model_name"),
        "test_set_metrics": _state.get("metrics"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SessionInput):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    cfg = _state["cfg"]
    row = payload.model_dump()

    timestamp = pd.Timestamp(row["charging_start_time"])
    row.update(derive_time_fields(timestamp))

    df = pd.DataFrame([row])
    df = add_time_features(df)
    df = add_physics_feature(df)

    feature_columns = cfg["numeric_features"] + cfg["categorical_features"]
    X = _state["preprocessor"].transform(df[feature_columns])

    if _state["model_name"] == "neural_net":
        prediction = float(_state["model"].predict(X, verbose=0).ravel()[0])
    else:
        prediction = float(_state["model"].predict(X)[0])

    return PredictionResponse(
        predicted_energy_kwh=round(max(prediction, 0.0), 3),
        model_used=_state["model_name"],
    )
