"""pca_project — PCA vs Deep Autoencoder Statistical Arbitrage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the project configuration from config.yaml.

    Args:
        path: Path to the YAML config file. Defaults to ``config.yaml`` at the
              repository root (two directories above this file).

    Returns:
        Parsed configuration as a nested Python dict.

    Raises:
        FileNotFoundError: If the config file does not exist at *path*.
    """
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        return yaml.safe_load(fh)
