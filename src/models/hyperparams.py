from pathlib import Path
from typing import Any

import yaml

from src import config


def load_hyperparameters(path: Path = config.MODELS_CONFIG) -> dict[str, Any]:
    """Read model hyperparameters from the YAML config."""
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter config not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping, got {type(data).__name__}")

    for key in ("random_state", "early_stopping_rounds", "models"):
        if key not in data:
            raise ValueError(f"{path} is missing required key: {key}")

    models = data["models"]

    if not isinstance(models, dict) or not models:
        raise ValueError(f"{path}: 'models' must be a non-empty mapping")

    for model_name, params in models.items():
        if not isinstance(params, dict):
            raise TypeError(f"{path}: model '{model_name}' must map to its parameters")

    return data
