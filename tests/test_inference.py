import pytest

from src.inference import load_model_bundle, predict_session
from src.utils.config import load_config, resolve_path


def _valid_session():
    return dict(
        vehicle_model="Tesla Model 3",
        battery_capacity_kwh=60.0,
        charger_type="DC Fast Charger",
        charging_start_time="2024-06-15T18:30:00",
        charging_duration_hours=0.45,
        charging_rate_kw=90.0,
        charging_cost_usd=16.8,
        soc_start_pct=20.0,
        soc_end_pct=80.0,
        distance_since_last_charge_km=200.0,
        temperature_c=20.0,
        vehicle_age_years=2.0,
        user_type="Long-Distance Traveler",
    )


def _artifacts_available() -> bool:
    cfg = load_config()
    artifacts_dir = resolve_path(cfg["artifacts"]["dir"])
    preprocessor = artifacts_dir / cfg["artifacts"]["preprocessor_file"]
    metrics = artifacts_dir / cfg["artifacts"]["metrics_file"]
    return preprocessor.exists() and metrics.exists()


def test_load_model_bundle_raises_clear_error_without_artifacts(tmp_path):
    cfg = load_config()
    cfg = dict(cfg)
    cfg["artifacts"] = dict(cfg["artifacts"])
    cfg["artifacts"]["dir"] = str(tmp_path / "nonexistent")

    with pytest.raises(RuntimeError, match="run_pipeline.py"):
        load_model_bundle(cfg)


@pytest.mark.skipif(
    not _artifacts_available(),
    reason="Run `python pipeline/run_pipeline.py` first to generate model artifacts",
)
def test_predict_session_matches_physics_estimate_closely():
    bundle = load_model_bundle()
    prediction = predict_session(_valid_session(), bundle)

    # battery_capacity * soc_change / 100 = 60 * 60 / 100 = 36; with ~0.9
    # charging efficiency the physically expected energy is ~40 kWh
    assert 30 < prediction < 50


@pytest.mark.skipif(
    not _artifacts_available(),
    reason="Run `python pipeline/run_pipeline.py` first to generate model artifacts",
)
def test_predict_session_is_never_negative():
    bundle = load_model_bundle()
    session = _valid_session()
    session["soc_start_pct"] = session["soc_end_pct"]  # zero SoC change

    prediction = predict_session(session, bundle)

    assert prediction >= 0
