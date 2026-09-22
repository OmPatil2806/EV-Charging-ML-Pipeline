# EV Charging Energy Prediction — ML Pipeline

Predicts energy consumption (kWh) for an EV charging session from session characteristics
(vehicle, charger type, time of day, battery capacity, state of charge, etc.), and compares
classical ML and deep learning models on the task.

> Status: under active development. This README is updated as each phase lands.

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
│   ├── models/                   # sklearn_models.py, neural_net.py, train.py
│   ├── evaluation/               # evaluate.py
│   └── utils/                    # config.py, logger.py
├── pipeline/run_pipeline.py      # end-to-end orchestration
├── models_artifacts/             # saved best model + preprocessing artifacts
├── notebooks/                    # EDA & experiments (readable narrative)
├── api/                          # FastAPI inference service
├── dashboard/                    # Streamlit app
└── tests/                        # unit tests
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
- [ ] Phase 8 — Dashboard (Streamlit)
- [ ] Phase 9 — Tests & documentation

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

## 8. Results

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
