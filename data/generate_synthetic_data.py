"""Generate a synthetic EV charging sessions dataset.

Unlike a purely random dataset, energy consumed here is derived from real
physical relationships (battery capacity, state-of-charge change, charging
efficiency, temperature derating) plus realistic noise, so downstream models
have genuine signal to learn. Missing values, out-of-range sensor readings,
and duplicate rows are injected deliberately so the cleaning phase has real
data-quality issues to handle.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.time_features import time_of_day_bucket  # noqa: E402

VEHICLE_MODELS = {
    "Tesla Model 3": {"battery_kwh": 60.0, "max_charge_rate_kw": 170.0, "km_per_kwh": 6.5},
    "Nissan Leaf": {"battery_kwh": 40.0, "max_charge_rate_kw": 50.0, "km_per_kwh": 5.5},
    "Chevy Bolt": {"battery_kwh": 65.0, "max_charge_rate_kw": 55.0, "km_per_kwh": 5.8},
    "BMW i3": {"battery_kwh": 42.0, "max_charge_rate_kw": 50.0, "km_per_kwh": 5.0},
    "Hyundai Kona": {"battery_kwh": 64.0, "max_charge_rate_kw": 77.0, "km_per_kwh": 6.2},
}

CHARGER_TYPES = {
    "Level 1": {"nominal_kw": 1.9, "price_per_kwh": 0.15},
    "Level 2": {"nominal_kw": 11.0, "price_per_kwh": 0.20},
    "DC Fast Charger": {"nominal_kw": 100.0, "price_per_kwh": 0.35},
}
CHARGER_TYPE_PROBS = [0.15, 0.55, 0.30]

USER_TYPES = ["Commuter", "Casual Driver", "Long-Distance Traveler"]

TIME_OF_DAY_COST_MULTIPLIER = {"Morning": 1.0, "Afternoon": 1.1, "Evening": 1.2, "Night": 0.85}


def generate_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    model_names = list(VEHICLE_MODELS.keys())
    vehicle_model = rng.choice(model_names, size=n)
    base_battery = np.array([VEHICLE_MODELS[m]["battery_kwh"] for m in vehicle_model])
    max_charge_rate = np.array([VEHICLE_MODELS[m]["max_charge_rate_kw"] for m in vehicle_model])
    km_per_kwh = np.array([VEHICLE_MODELS[m]["km_per_kwh"] for m in vehicle_model])
    battery_capacity_kwh = np.clip(base_battery + rng.normal(0, base_battery * 0.03, n), 20, None)

    charger_names = list(CHARGER_TYPES.keys())
    charger_type = rng.choice(charger_names, size=n, p=CHARGER_TYPE_PROBS)
    nominal_kw = np.array([CHARGER_TYPES[c]["nominal_kw"] for c in charger_type])
    price_per_kwh = np.array([CHARGER_TYPES[c]["price_per_kwh"] for c in charger_type])
    charging_rate_kw = np.minimum(nominal_kw * rng.uniform(0.85, 1.05, n), max_charge_rate)

    user_type = rng.choice(USER_TYPES, size=n)
    vehicle_age_years = np.round(rng.uniform(0, 10, n), 1)
    temperature_c = np.clip(rng.normal(15, 12, n), -15, 45)

    soc_start_pct = rng.uniform(5, 70, n)
    soc_gain = rng.uniform(10, 60, n)
    soc_end_pct = np.clip(soc_start_pct + soc_gain, 0, 100)
    soc_change_pct = soc_end_pct - soc_start_pct

    temp_penalty = np.where(temperature_c < 5, (5 - temperature_c) * 0.004, 0.0)
    temp_penalty += np.where(temperature_c > 35, (temperature_c - 35) * 0.003, 0.0)
    efficiency = np.clip(0.92 - 0.002 * vehicle_age_years - temp_penalty, 0.75, 0.95)

    energy_to_battery_kwh = battery_capacity_kwh * soc_change_pct / 100
    energy_consumed_kwh = np.clip(
        energy_to_battery_kwh / efficiency + rng.normal(0, 0.5, n), 0.1, None
    )

    charging_duration_hours = np.clip(
        energy_consumed_kwh / charging_rate_kw * rng.uniform(0.95, 1.15, n), 0.05, None
    )

    depletion_pct = 100 - soc_start_pct
    distance_since_last_charge_km = np.clip(
        battery_capacity_kwh * depletion_pct / 100 * km_per_kwh * rng.uniform(0.7, 1.3, n),
        0,
        None,
    )

    start_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n), unit="D"
    ) + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    charging_start_time = pd.Series(start_dates)
    time_of_day = charging_start_time.dt.hour.map(time_of_day_bucket).to_numpy()
    day_of_week = charging_start_time.dt.day_name().to_numpy()

    cost_multiplier = np.array([TIME_OF_DAY_COST_MULTIPLIER[t] for t in time_of_day])
    charging_cost_usd = np.clip(
        energy_consumed_kwh * price_per_kwh * cost_multiplier + rng.normal(0, 0.3, n), 0, None
    )

    user_pool = [f"USER_{i:04d}" for i in range(max(1, n // 10))]
    user_id = rng.choice(user_pool, size=n)
    session_id = [f"SESSION_{i:06d}" for i in range(n)]

    df = pd.DataFrame(
        {
            "session_id": session_id,
            "user_id": user_id,
            "vehicle_model": vehicle_model,
            "battery_capacity_kwh": battery_capacity_kwh.round(2),
            "charger_type": charger_type,
            "charging_start_time": charging_start_time,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "charging_duration_hours": charging_duration_hours.round(3),
            "charging_rate_kw": charging_rate_kw.round(2),
            "charging_cost_usd": charging_cost_usd.round(2),
            "soc_start_pct": soc_start_pct.round(1),
            "soc_end_pct": soc_end_pct.round(1),
            "distance_since_last_charge_km": distance_since_last_charge_km.round(1),
            "temperature_c": temperature_c.round(1),
            "vehicle_age_years": vehicle_age_years,
            "user_type": user_type,
            "energy_consumed_kwh": energy_consumed_kwh.round(3),
        }
    )
    return df


def inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    missing_rate = cfg["data"]["missing_value_rate"]
    outlier_rate = cfg["data"]["outlier_rate"]

    for column in ["charging_rate_kw", "distance_since_last_charge_km", "energy_consumed_kwh"]:
        mask = rng.random(n) < missing_rate
        df.loc[mask, column] = np.nan

    n_outliers = int(n * outlier_rate)
    if n_outliers > 0:
        soc_start_idx = rng.choice(n, size=n_outliers, replace=False)
        df.loc[soc_start_idx, "soc_start_pct"] = rng.choice(
            [-rng.uniform(1, 15), rng.uniform(101, 160)], size=n_outliers
        )
        soc_end_idx = rng.choice(n, size=n_outliers, replace=False)
        df.loc[soc_end_idx, "soc_end_pct"] = rng.choice(
            [-rng.uniform(1, 15), rng.uniform(101, 180)], size=n_outliers
        )

    n_duplicates = max(1, int(n * 0.005))
    duplicate_rows = df.sample(n=n_duplicates, random_state=int(rng.integers(0, 1_000_000)))
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

    return df


def main():
    cfg = load_config()
    seed = cfg["random_seed"]
    n = cfg["data"]["n_sessions"]

    df = generate_dataset(n=n, seed=seed)
    df = inject_data_quality_issues(df, rng=np.random.default_rng(seed + 1), cfg=cfg)

    raw_path = resolve_path(cfg["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)

    print(f"Generated {len(df)} rows -> {raw_path}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(
        "Correlation(energy_consumed_kwh, battery_capacity_kwh * soc_change): "
        f"{np.corrcoef(df['energy_consumed_kwh'].fillna(0), (df['soc_end_pct'] - df['soc_start_pct']) * df['battery_capacity_kwh'])[0, 1]:.4f}"
    )


if __name__ == "__main__":
    main()
