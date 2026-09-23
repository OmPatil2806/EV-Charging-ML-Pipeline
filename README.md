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
  dashboard/app.py  (Streamlit: EDA + model comparison + live prediction)
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
dashboard/app.py           # Streamlit app
notebooks/                 # executed EDA + model comparison notebook
tests/                     # 40 tests across every module
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python pipeline/run_pipeline.py      # generate data, train, evaluate, save best model
streamlit run dashboard/app.py       # explore data, compare models, try live predictions
pytest tests/ -v                     # 40 tests
```

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
