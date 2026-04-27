"""Experiment result schema and result persistence utilities."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Structured container for a single grid-search experiment result.

    Attributes:
        model_type: ``"pca"`` or ``"autoencoder"``.
        experiment_id: Unique identifier string.
        timestamp: ISO-format timestamp string.
        n_factors: Number of factors / bottleneck size.
        zscore_entry: Entry Z-score threshold.
        zscore_exit: Exit Z-score threshold.
        depth: Hidden-layer depth (AE only; None for PCA).
        activation: Activation function name (AE only; None for PCA).
        variance_explained: Fraction of variance explained (PCA only).
        final_val_loss: Best validation MSE loss (AE only).
        sharpe_with_costs: Sharpe ratio after transaction costs.
        max_drawdown_with_costs: Maximum drawdown after costs.
        hit_ratio_with_costs: Hit ratio after costs.
        annualized_return_with_costs: Annualized return after costs.
        annualized_turnover_with_costs: Annualized turnover after costs.
        sharpe_without_costs: Sharpe ratio before transaction costs.
        max_drawdown_without_costs: Maximum drawdown before costs.
        hit_ratio_without_costs: Hit ratio before costs.
        annualized_return_without_costs: Annualized return before costs.
        annualized_turnover_without_costs: Annualized turnover before costs.
    """

    # Identification
    model_type: str
    experiment_id: str
    timestamp: str

    # Hyperparameters
    n_factors: int
    zscore_entry: float
    zscore_exit: float
    depth: int | None
    activation: str | None

    # Model quality
    variance_explained: float | None
    final_val_loss: float | None

    # Performance with transaction costs
    sharpe_with_costs: float
    max_drawdown_with_costs: float
    hit_ratio_with_costs: float
    annualized_return_with_costs: float
    annualized_turnover_with_costs: float

    # Performance without transaction costs
    sharpe_without_costs: float
    max_drawdown_without_costs: float
    hit_ratio_without_costs: float
    annualized_return_without_costs: float
    annualized_turnover_without_costs: float


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _results_path(filename: str, config: dict[str, Any]) -> Path:
    return Path(config["data"]["results_dir"]) / filename


def save_results(results: Any, filename: str, config: dict[str, Any]) -> None:
    """Serialize a results object to ``data/results/<filename>.pkl``.

    If the object is (or contains) a DataFrame, companion ``.csv`` files are
    also written for easy inspection without Python.

    Args:
        results: Any serializable Python object.
        filename: Base filename (e.g. ``"pca_grid_search.pkl"``).
        config: Project configuration dict.
    """
    path = _results_path(filename, config)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as fh:
        pickle.dump(results, fh)
    logger.info("Saved results to %s", path)

    # Companion CSV for DataFrames
    if isinstance(results, pd.DataFrame):
        csv_path = path.with_suffix(".csv")
        results.to_csv(csv_path)
        logger.debug("Companion CSV saved to %s", csv_path)
    elif isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, pd.DataFrame):
                csv_path = path.parent / f"{path.stem}_{k}.csv"
                v.to_csv(csv_path)
                logger.debug("Companion CSV saved to %s", csv_path)


def load_results(filename: str, config: dict[str, Any]) -> Any:
    """Load a previously saved results object from ``data/results/<filename>.pkl``.

    Args:
        filename: Base filename (e.g. ``"pca_grid_search.pkl"``).
        config: Project configuration dict.

    Returns:
        The deserialized Python object.
    """
    path = _results_path(filename, config)
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    logger.info("Loaded results from %s", path)
    return obj


def results_exist(filename: str, config: dict[str, Any]) -> bool:
    """Check whether a results file exists on disk.

    Args:
        filename: Base filename (e.g. ``"data_splits.pkl"``).
        config: Project configuration dict.

    Returns:
        ``True`` if the file exists.
    """
    return _results_path(filename, config).exists()
