"""End-to-end pipeline: generate data (if needed) -> clean -> engineer
features -> train all models -> evaluate -> save the best model's artifacts.

Usage:
    python pipeline/run_pipeline.py
    python pipeline/run_pipeline.py --regenerate-data
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_synthetic_data import main as generate_synthetic_data  # noqa: E402
from src.evaluation.evaluate import evaluate_and_save  # noqa: E402
from src.features.feature_engineering import build_feature_dataset  # noqa: E402
from src.models.train import train_all_models  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full EV charging ML pipeline end-to-end.")
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate the synthetic raw dataset even if it already exists.",
    )
    return parser.parse_args()


def step(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_pipeline(regenerate_data: bool = False) -> str:
    cfg = load_config()
    start = time.time()

    raw_path = resolve_path(cfg["data"]["raw_path"])
    if regenerate_data or not raw_path.exists():
        step("Step 1/4: Generating synthetic data")
        generate_synthetic_data()
    else:
        step("Step 1/4: Using existing raw dataset")
        print(f"Found {raw_path} (pass --regenerate-data to rebuild it)")

    step("Step 2/4: Cleaning + feature engineering")
    df = build_feature_dataset(cfg)
    processed_path = resolve_path(cfg["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved feature-engineered dataset ({df.shape[0]} rows, {df.shape[1]} cols) -> {processed_path}")

    step("Step 3/4: Training all models")
    results = train_all_models(cfg, df=df)

    step("Step 4/4: Evaluating and saving the best model")
    best_name = evaluate_and_save(results, cfg)

    elapsed = time.time() - start
    print(f"\nPipeline complete in {elapsed:.1f}s. Best model: {best_name}")
    return best_name


def main():
    args = parse_args()
    run_pipeline(regenerate_data=args.regenerate_data)


if __name__ == "__main__":
    main()
