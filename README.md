# EV Charging Energy Prediction — ML Pipeline

Predicts energy consumption (kWh) for an EV charging session from session characteristics
(vehicle, charger type, time of day, battery capacity, state of charge, etc.), and compares
classical ML and deep learning models on the task.

> Status: complete. All 9 build phases below are implemented, tested, and reproducible end-to-end.

## 1. Business Problem

Charging infrastructure providers need to understand and forecast energy demand per session to
optimize station usage, manage grid load, and improve user experience. This project builds an
end-to-end pipeline — data generation, cleaning, feature engineering, model training, evaluation,
and serving — to predict per-session energy consumption.

## 2. Architecture

```
generate_synthetic_data.py
        │  (physics-based signal: Energy ≈ f(Battery, ΔSoC, efficiency) + noise)
        ▼
   data/raw/ev_charging_sessions.csv
        │
        ▼
  ingestion.py → cleaning.py → feature_engineering.py
        │
        ▼
   data/processed/ev_charging_features.csv
        │
        ▼
      train.py  ──trains──▶  LinearRegression / Ridge / RandomForest /
        │                    GradientBoosting / Keras MLP
        ▼
    evaluate.py  (RMSE / MAE / R² leaderboard, feature importances, residual plots)
        │
        ▼
  best model + preprocessor  →  models_artifacts/
        │
        ▼
  src/inference.py  (shared scoring logic: load bundle, build features, predict)
        │
        ├──▶ api/main.py       (FastAPI: POST /predict)
        └──▶ dashboard/app.py  (Streamlit: EDA + model comparison + live prediction)
```

## 3. Project Structure

```
EV-Charging-ML-Pipeline/
├── config/config.yaml            # all paths, feature lists, hyperparameters
├── data/
│   ├── generate_synthetic_data.py
│   ├── raw/                      # generated raw dataset
│   └── processed/                # cleaned + feature-engineered dataset
├── src/
│   ├── data/                     # ingestion.py, cleaning.py
│   ├── features/                 # feature_engineering.py
│   ├── models/                   # sklearn_models.py, neural_net.py, train.py, preprocessing.py
│   ├── evaluation/                # evaluate.py
│   ├── inference.py               # shared scoring logic used by both api/ and dashboard/
│   └── utils/                    # config.py, time_features.py
├── pipeline/run_pipeline.py      # end-to-end orchestration
├── models_artifacts/             # saved best model + preprocessing artifacts
├── notebooks/eda_and_experiments.ipynb  # executed EDA + model comparison narrative
├── api/                          # FastAPI inference service
├── dashboard/app.py              # Streamlit app (EDA, model comparison, live prediction)
└── tests/                        # 40 unit/integration tests across every module above
```

## 4. Build Phases

- [x] Phase 0 — Repo scaffolding
- [x] Phase 1 — Synthetic data generation
- [x] Phase 2 — Ingestion & cleaning
- [x] Phase 3 — Feature engineering
- [x] Phase 4 — Model training (classical ML + Keras MLP)
- [x] Phase 5 — Evaluation & model selection
- [x] Phase 6 — Pipeline orchestration
- [x] Phase 7 — Inference API (FastAPI)
- [x] Phase 8 — Dashboard (Streamlit)
- [x] Phase 9 — Tests & documentation

## 5. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Running the Pipeline

Run everything end-to-end (data generation, if needed → cleaning → feature engineering →
training all 5 models → evaluation → saving the best model) with one command:

```bash
python pipeline/run_pipeline.py
```

The raw dataset is only regenerated if it doesn't already exist. To force a fresh synthetic
dataset (new random sessions):

```bash
python pipeline/run_pipeline.py --regenerate-data
```

Each phase can also be run individually:

```bash
python data/generate_synthetic_data.py     # Phase 1
python -m src.data.ingestion               # Phase 2 (inspect raw data quality)
python -m src.data.cleaning                # Phase 2
python -m src.features.feature_engineering # Phase 3
python -m src.models.train                 # Phase 4
python -m src.evaluation.evaluate          # Phase 5
```

Note: exact metrics vary slightly between runs (random train/test split, neural net
initialization) — Gradient Boosting and the Neural Net consistently trade the top spot,
both well ahead of the linear baselines.

## 7. Inference API

Once the pipeline has produced `models_artifacts/` (run Phase 6 first), serve predictions with:

```bash
uvicorn api.main:app --reload
```

- `GET /health` — service + model-loaded status
- `GET /` — which model is currently serving, and its test-set metrics
- `POST /predict` — predict energy consumed (kWh) for a session

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_model": "Tesla Model 3",
    "battery_capacity_kwh": 60.0,
    "charger_type": "DC Fast Charger",
    "charging_start_time": "2024-06-15T18:30:00",
    "charging_duration_hours": 0.45,
    "charging_rate_kw": 90.0,
    "charging_cost_usd": 16.8,
    "soc_start_pct": 20.0,
    "soc_end_pct": 80.0,
    "distance_since_last_charge_km": 200.0,
    "temperature_c": 20.0,
    "vehicle_age_years": 2.0,
    "user_type": "Long-Distance Traveler"
  }'
# {"predicted_energy_kwh": 40.51, "model_used": "neural_net"}
```

The API derives `day_of_week`/`time_of_day` from `charging_start_time` and reuses the exact same
`add_time_features`/`add_physics_feature` functions from Phase 3 to build engineered features — so
there's a single source of truth for feature construction between training and serving, and the
saved preprocessor guarantees identical scaling/encoding at inference time.

Interactive API docs (Swagger UI): `http://127.0.0.1:8000/docs`

## 8. Dashboard

An interactive Streamlit dashboard sits on top of the same artifacts as the API:

```bash
streamlit run dashboard/app.py
```

- **Data Exploration** — dataset size, feature distributions, correlation with the target
  (diverging bar chart), and categorical session counts.
- **Model Comparison** — the test-set leaderboard as a table and bar chart, plus the residual
  plots from Phase 5.
- **Live Prediction** — a form for a charging session's characteristics that calls
  `src/inference.py` (the same code path the API uses) and shows the prediction alongside a
  physics-based sanity estimate (`battery capacity × ΔSoC / efficiency`) for comparison.

## 9. Results

Test-set performance (981 held-out sessions), predicting `energy_consumed_kwh`:

| Model | RMSE (kWh) | MAE (kWh) | R² |
|---|---|---|---|
| **Gradient Boosting** (selected) | **1.29** | **0.59** | **0.980** |
| Neural Net (Keras MLP) | 1.45 | 0.93 | 0.975 |
| Random Forest | 1.56 | 0.66 | 0.971 |
| Linear Regression | 2.24 | 1.63 | 0.940 |
| Ridge | 2.24 | 1.63 | 0.940 |

Gradient Boosting was selected automatically (lowest test RMSE) and saved to `models_artifacts/`.
The dominant predictor for every tree-based model is `expected_energy_kwh` (the physics-based
feature: Battery Capacity × SoC Change / 100), accounting for ~95% of feature importance — confirming
the synthetic dataset carries a real, learnable physical relationship, unlike the original notebook's
dataset (R² ≈ 0.01 there, on unrelated/near-random data).

Reproduce with:
```bash
python -m src.evaluation.evaluate
```

A full narrative walkthrough of this analysis — with the same plots, rendered inline — is in
[`notebooks/eda_and_experiments.ipynb`](notebooks/eda_and_experiments.ipynb).

## 10. Testing

40 tests across every module (`pytest tests/`):

| File | Covers |
|---|---|
| `test_generate_synthetic_data.py` | synthetic data generation: schema, physical plausibility, seed determinism, injected data-quality issues |
| `test_ingestion.py` | raw data summary stats |
| `test_cleaning.py` | dedup, missing-target drop, median imputation, range capping |
| `test_feature_engineering.py` | time features, SoC change, the physics feature, identifier column drop |
| `test_models.py` | preprocessing pipeline (incl. leakage: unseen test categories), model grid, Keras MLP construction/training |
| `test_train.py` | `train_all_models()` end-to-end on a small synthetic dataset with known signal |
| `test_evaluate.py` | metric computation, best-model selection logic |
| `test_inference.py` | shared scoring path, missing-artifacts error message, prediction sanity bounds |
| `test_api.py` | request schema validation, live `/predict` and `/health` (skipped if artifacts aren't built yet) |

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests that need trained model artifacts (`test_inference.py`, `test_api.py`) skip cleanly with a
clear reason if `python pipeline/run_pipeline.py` hasn't been run yet, rather than failing.

## 11. Limitations & Future Work

- **Synthetic data.** The dataset is generated, not real-world EV telemetry. The physical
  relationship (Battery Capacity × ΔSoC / efficiency) is realistic, but real charging data would
  have messier, weaker correlations, more edge cases, and likely additional unmodeled factors
  (battery chemistry, charging curve tapering, grid conditions).
- **Neural net run-to-run variance.** The Keras MLP's exact metrics shift slightly between runs
  even with a fixed seed (TensorFlow doesn't guarantee full operation-level determinism on CPU).
  Gradient Boosting and the neural net consistently trade the top spot; both consistently beat the
  linear baselines.
- **No experiment tracking (e.g. MLflow).** Metrics are persisted to `metrics.json` per run, but
  there's no run history/comparison UI — a reasonable next step if this were run repeatedly over
  time with changing data or hyperparameters.
- **No containerization/CI.** The project runs from a local virtualenv; a `Dockerfile` and a CI
  workflow (running `pytest` on push) would be the natural next additions for a deployed version.
- **Single-session prediction only.** The API/dashboard predict one session at a time; a batch
  `/predict-many` endpoint would be a small, useful extension.
