"""FastAPI inference service: loads the best trained model + preprocessor
(saved by src/evaluation/evaluate.py) and serves energy consumption
predictions for a charging session.

Run with:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import PredictionResponse, SessionInput
from src.inference import load_model_bundle, predict_session

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["bundle"] = load_model_bundle()
    yield


app = FastAPI(
    title="EV Charging Energy Prediction API",
    description="Predicts energy consumed (kWh) for an EV charging session.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    bundle = _state.get("bundle")
    return {
        "service": "EV Charging Energy Prediction API",
        "model": bundle["model_name"] if bundle else None,
        "test_set_metrics": bundle["leaderboard"][bundle["model_name"]] if bundle else None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "bundle" in _state}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SessionInput):
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    session = payload.model_dump(mode="json")
    prediction = predict_session(session, bundle)

    return PredictionResponse(
        predicted_energy_kwh=round(prediction, 3),
        model_used=bundle["model_name"],
    )
