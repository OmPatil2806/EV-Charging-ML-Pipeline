import pytest
from pydantic import ValidationError

from api.schemas import SessionInput
from src.utils.config import load_config, resolve_path


def _valid_payload():
    return dict(
        vehicle_model="Tesla Model 3",
        battery_capacity_kwh=60.0,
        charger_type="Level 2",
        charging_start_time="2024-06-15T18:30:00",
        charging_duration_hours=2.5,
        charging_rate_kw=11.0,
        charging_cost_usd=8.5,
        soc_start_pct=25.0,
        soc_end_pct=80.0,
        distance_since_last_charge_km=180.0,
        temperature_c=18.0,
        vehicle_age_years=2.0,
        user_type="Commuter",
    )


def test_session_input_accepts_valid_payload():
    session = SessionInput(**_valid_payload())
    assert session.vehicle_model == "Tesla Model 3"
    assert session.soc_end_pct == 80.0


def test_session_input_rejects_out_of_range_soc():
    payload = _valid_payload()
    payload["soc_start_pct"] = 150.0
    with pytest.raises(ValidationError):
        SessionInput(**payload)


def test_session_input_rejects_negative_battery_capacity():
    payload = _valid_payload()
    payload["battery_capacity_kwh"] = -5.0
    with pytest.raises(ValidationError):
        SessionInput(**payload)


def _artifacts_available() -> bool:
    cfg = load_config()
    artifacts_dir = resolve_path(cfg["artifacts"]["dir"])
    preprocessor = artifacts_dir / cfg["artifacts"]["preprocessor_file"]
    metrics = artifacts_dir / cfg["artifacts"]["metrics_file"]
    return preprocessor.exists() and metrics.exists()


@pytest.mark.skipif(
    not _artifacts_available(),
    reason="Run `python pipeline/run_pipeline.py` first to generate model artifacts",
)
def test_predict_endpoint_returns_a_positive_prediction():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_energy_kwh"] > 0
    assert "model_used" in body


@pytest.mark.skipif(
    not _artifacts_available(),
    reason="Run `python pipeline/run_pipeline.py` first to generate model artifacts",
)
def test_health_endpoint_reports_model_loaded():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
