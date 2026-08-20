"""
Utility functions for the Breast Cancer Prediction project.
Handles config loading, artifact save/load, and logging setup.
"""

import os
import yaml
import joblib
import logging
from typing import Any, Dict


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with consistent formatting."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_artifact(obj: Any, filepath: str) -> None:
    """Save a Python object using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    logging.info(f"Artifact saved: {filepath}")


def load_artifact(filepath: str) -> Any:
    """Load a Python object using joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    obj = joblib.load(filepath)
    logging.info(f"Artifact loaded: {filepath}")
    return obj


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Create all necessary directories from config."""
    dirs = [
        config["paths"]["model_dir"],
        config["paths"]["output_dir"],
        config["paths"]["plot_dir"],
        config["paths"]["metrics_dir"],
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Directory ensured: {d}")
