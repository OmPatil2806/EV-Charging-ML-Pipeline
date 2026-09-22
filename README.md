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
  best model + scaler + encoder  →  models_artifacts/
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
- [ ] Phase 4 — Model training (classical ML + Keras MLP)
- [ ] Phase 5 — Evaluation & model selection
- [ ] Phase 6 — Pipeline orchestration
- [ ] Phase 7 — Inference API (FastAPI)
- [ ] Phase 8 — Dashboard (Streamlit)
- [ ] Phase 9 — Tests & documentation

## 5. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Running the Pipeline

```bash
python pipeline/run_pipeline.py
```

(Filled in once Phase 6 lands.)

## 7. Results

(Filled in once Phase 5 lands — model comparison table and chosen best model.)
