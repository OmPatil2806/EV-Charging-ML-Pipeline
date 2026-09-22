"""Load the project's central YAML config and resolve paths relative to the repo root."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path
