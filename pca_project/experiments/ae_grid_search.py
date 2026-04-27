"""Grid search over autoencoder hyperparameters."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from itertools import product
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from pca_project.backtesting.engine import run_full_backtest
from pca_project.experiments import ExperimentResult
from pca_project.factors.autoencoder_model import AutoencoderModel
from pca_project.metrics.performance import PerformanceAnalyzer

logger = logging.getLogger(__name__)


def _run_single_ae(
    bottleneck: int,
    depth: int,
    activation: str,
    zscore_entry: float,
    zscore_exit: float,
    data: dict,
    config: dict[str, Any],
) -> ExperimentResult:
    """Run one autoencoder configuration and return its ExperimentResult.

    Args:
        bottleneck: Latent dimension.
        depth: Number of hidden layers per side.
        activation: Activation function name.
        zscore_entry: Z-score entry threshold.
        zscore_exit: Z-score exit threshold.
        data: Data dict from DataPreprocessor.run().
        config: Project configuration dict.

    Returns:
        Populated ExperimentResult.
    """
    model = AutoencoderModel(config, bottleneck=bottleneck, depth=depth, activation=activation)
    model.fit(data["train_std"], data["val_std"])

    bt = run_full_backtest(
        model,
        data["test_raw"],
        data["test_std"],
        config,
        zscore_entry=zscore_entry,
        zscore_exit=zscore_exit,
    )

    analyzer = PerformanceAnalyzer(config)
    m_with = analyzer.compute_all(
        bt["with_costs"]["daily_returns"], bt["with_costs"]["weights"]
    )
    m_without = analyzer.compute_all(
        bt["without_costs"]["daily_returns"], bt["without_costs"]["weights"]
    )

    return ExperimentResult(
        model_type="autoencoder",
        experiment_id=f"ae_b{bottleneck}_d{depth}_{activation}_e{zscore_entry}_x{zscore_exit}",
        timestamp=datetime.utcnow().isoformat(),
        n_factors=bottleneck,
        zscore_entry=zscore_entry,
        zscore_exit=zscore_exit,
        depth=depth,
        activation=activation,
        variance_explained=None,
        final_val_loss=model.final_val_loss_,
        sharpe_with_costs=m_with["sharpe_ratio"],
        max_drawdown_with_costs=m_with["maximum_drawdown"],
        hit_ratio_with_costs=m_with["hit_ratio"],
        annualized_return_with_costs=m_with["annualized_return"],
        annualized_turnover_with_costs=m_with["annualized_turnover"],
        sharpe_without_costs=m_without["sharpe_ratio"],
        max_drawdown_without_costs=m_without["maximum_drawdown"],
        hit_ratio_without_costs=m_without["hit_ratio"],
        annualized_return_without_costs=m_without["annualized_return"],
        annualized_turnover_without_costs=m_without["annualized_turnover"],
    )


class AEGridSearch:
    """Grid search over autoencoder hyperparameters.

    When ``full_grid=False`` (default), only the default hyperparameter values
    from config are tested — a fast first pass. When ``full_grid=True``, the
    full cross-product is evaluated (warns about compute time).

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, data: dict, full_grid: bool = False, verbose: bool = True) -> pd.DataFrame:
        """Execute the AE grid search and return a results DataFrame.

        Args:
            data: Data dict from DataPreprocessor.run().
            full_grid: If True, test the full cross-product of all grids.
                       If False, only test default hyperparameter values.
            verbose: Log progress and summary.

        Returns:
            DataFrame with one row per configuration, sorted by Sharpe ratio
            (with costs) descending.
        """
        ae_cfg = self.config["autoencoder"]
        sig_cfg = self.config["signals"]

        if full_grid:
            bottleneck_vals = ae_cfg["bottleneck_grid"]
            depth_vals = ae_cfg["depth_grid"]
            act_vals = ae_cfg["activation_grid"]
            logger.warning(
                "AE full grid search: %d × %d × %d × %d × %d = %d configurations. "
                "This may take a long time.",
                len(bottleneck_vals),
                len(depth_vals),
                len(act_vals),
                len(sig_cfg["zscore_entry_grid"]),
                len(sig_cfg["zscore_exit_grid"]),
                len(bottleneck_vals) * len(depth_vals) * len(act_vals)
                * len(sig_cfg["zscore_entry_grid"]) * len(sig_cfg["zscore_exit_grid"]),
            )
            entry_grid = sig_cfg["zscore_entry_grid"]
            exit_grid = sig_cfg["zscore_exit_grid"]
        else:
            bottleneck_vals = [ae_cfg["default_bottleneck"]]
            depth_vals = ae_cfg["depth_grid"]          # vary depth and activation
            act_vals = ae_cfg["activation_grid"]        # with default bottleneck/thresholds
            entry_grid = [sig_cfg["default_zscore_entry"]]
            exit_grid = [sig_cfg["default_zscore_exit"]]
            logger.info(
                "AE grid search (default bottleneck=%d): %d depth × %d activation = %d configs",
                ae_cfg["default_bottleneck"],
                len(depth_vals),
                len(act_vals),
                len(depth_vals) * len(act_vals),
            )

        configs = list(product(bottleneck_vals, depth_vals, act_vals, entry_grid, exit_grid))
        n_configs = len(configs)
        logger.info("AE grid search: %d configurations", n_configs)
        t0 = time.perf_counter()

        # AE training is CPU-bound and not thread-safe with PyTorch; use sequential
        # processing to avoid deadlocks (n_jobs=1 for AE grid search)
        results: list[ExperimentResult] = []
        for b, d, act, e, x in configs:
            result = _run_single_ae(b, d, act, e, x, data, self.config)
            results.append(result)

        elapsed = time.perf_counter() - t0
        df = pd.DataFrame([r.__dict__ for r in results])
        df = df.sort_values("sharpe_with_costs", ascending=False).reset_index(drop=True)

        if verbose and len(df) > 0:
            best = df.iloc[0]
            logger.info(
                "AE grid search complete in %.1fs. %d configs tested. "
                "Best Sharpe (with costs) = %.4f at bottleneck=%d, depth=%d, act=%s",
                elapsed,
                n_configs,
                best["sharpe_with_costs"],
                best["n_factors"],
                best["depth"],
                best["activation"],
            )
        return df

    def get_best_config(
        self, results_df: pd.DataFrame, metric: str = "sharpe_with_costs"
    ) -> dict[str, Any]:
        """Extract the hyperparameter dict for the best configuration.

        Args:
            results_df: DataFrame returned by ``run()``.
            metric: Column name to optimize (default: ``sharpe_with_costs``).

        Returns:
            Dict with keys: bottleneck, depth, activation, zscore_entry, zscore_exit.
        """
        best = results_df.loc[results_df[metric].idxmax()]
        return {
            "bottleneck": int(best["n_factors"]),
            "depth": int(best["depth"]),
            "activation": str(best["activation"]),
            "zscore_entry": float(best["zscore_entry"]),
            "zscore_exit": float(best["zscore_exit"]),
        }
