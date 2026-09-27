# EV Charging Energy Prediction

An end-to-end ML pipeline that predicts energy consumption (kWh) for an EV charging session,
comparing 5 classical ML and deep learning models. **R² ≈ 0.98** on held-out test data.

## Architecture

```
generate_synthetic_data.py  (physics-based signal: Energy ≈ f(Battery, ΔSoC, efficiency) + noise)
        ▼
  ingestion → cleaning → feature engineering
        ▼
  train 5 models: LinearRegression / Ridge / RandomForest / GradientBoosting / Keras MLP
        ▼
  evaluate → auto-select best model → models_artifacts/
        ▼
  dashboard/  (Streamlit, layered: pages -> components -> services -> src/)
```

## Project Structure

```
config/config.yaml        # paths, features, hyperparameters
data/                      # synthetic data generator + raw/processed data
src/
├── data/                  # ingestion, cleaning
├── features/               # feature engineering
├── models/                # preprocessing, training, all 5 models
├── evaluation/             # leaderboard, feature importances, residual plots
└── inference.py            # shared scoring logic
pipeline/run_pipeline.py   # one-command end-to-end run
dashboard/
├── app.py                 # entrypoint: navigation wiring only
├── pages/                  # Overview, Data Exploration, Model Comparison, Live Prediction
├── components/             # reusable render functions (charts, KPI cards, form, sidebar)
├── services/                # cached data/model access - the only layer touching disk
├── theme/                  # validated chart color tokens
└── state.py                 # typed session-state accessors (filters, prediction history)
notebooks/                 # executed EDA + model comparison notebook
tests/                     # 44 tests across every module, incl. dashboard smoke tests
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python pipeline/run_pipeline.py      # generate data, train, evaluate, save best model
streamlit run dashboard/app.py       # explore data, compare models, try live predictions
pytest tests/ -v                     # 44 tests
```

## Dashboard

`streamlit run dashboard/app.py` — 4 pages, sidebar navigation:

- **Overview** — KPIs (sessions, best model, R², RMSE) and pipeline status at a glance.
- **Data Exploration** — feature distributions, correlation with target, category counts; filterable by vehicle model / charger type from the sidebar.
- **Model Comparison** — test-set leaderboard, RMSE chart, residual plots.
- **Live Prediction** — score a hypothetical session, with a running history of recent tries.

## Results

Test-set performance (981 held-out sessions):

| Model | RMSE (kWh) | MAE (kWh) | R² |
|---|---|---|---|
| **Gradient Boosting** (selected) | **1.29** | **0.59** | **0.980** |
| Neural Net (Keras MLP) | 1.45 | 0.93 | 0.975 |
| Random Forest | 1.56 | 0.66 | 0.971 |
| Linear Regression | 2.24 | 1.63 | 0.940 |
| Ridge | 2.24 | 1.63 | 0.940 |

The dominant predictor is `expected_energy_kwh`, a physics-engineered feature (Battery Capacity ×
SoC Change / 100) — ~95% of feature importance in every tree model. That's the difference between
this pipeline and a naive one: the synthetic data carries a real, learnable physical relationship
instead of near-random noise.

Full narrative with inline plots: [`notebooks/eda_and_experiments.ipynb`](notebooks/eda_and_experiments.ipynb)

## Limitations

- Synthetic data, not real-world telemetry — real data would be noisier.
- No experiment tracking, CI, or containerization yet.
- Neural net metrics vary slightly run-to-run (TensorFlow CPU non-determinism).
