"""Grid search over PCA hyperparameters (n_factors × zscore thresholds)."""

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
from pca_project.factors.pca_model import PCAModel
from pca_project.metrics.performance import PerformanceAnalyzer

logger = logging.getLogger(__name__)


def _run_single_pca(
    k: int,
    zscore_entry: float,
    zscore_exit: float,
    data: dict,
    config: dict[str, Any],
) -> ExperimentResult:
    """Run one PCA configuration and return its ExperimentResult.

    Args:
        k: Number of PCA factors.
        zscore_entry: Z-score entry threshold.
        zscore_exit: Z-score exit threshold.
        data: Data dict from DataPreprocessor.run().
        config: Project configuration dict.

    Returns:
        Populated ExperimentResult.
    """
    model = PCAModel(config, n_factors=k)
    model.fit(data["train_std"])
    summary = model.get_variance_explained_summary()

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
        model_type="pca",
        experiment_id=f"pca_k{k}_e{zscore_entry}_x{zscore_exit}",
        timestamp=datetime.utcnow().isoformat(),
        n_factors=k,
        zscore_entry=zscore_entry,
        zscore_exit=zscore_exit,
        depth=None,
        activation=None,
        variance_explained=summary["cumulative_variance_explained"],
        final_val_loss=None,
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


class PCAGridSearch:
    """Exhaustive grid search over PCA hyperparameters.

    Tests all combinations of ``n_factors × zscore_entry × zscore_exit``
    specified in ``config``. Runs configurations in parallel using joblib.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, data: dict, verbose: bool = True) -> pd.DataFrame:
        """Execute the grid search and return a results DataFrame.

        Args:
            data: Data dict from DataPreprocessor.run().
            verbose: Log progress and summary statistics.

        Returns:
            DataFrame with one row per configuration, sorted by Sharpe ratio
            (with costs) descending.
        """
        pca_cfg = self.config["pca"]
        sig_cfg = self.config["signals"]
        n_factors_grid = pca_cfg["n_factors_grid"]
        entry_grid = sig_cfg["zscore_entry_grid"]
        exit_grid = sig_cfg["zscore_exit_grid"]

        configs = list(product(n_factors_grid, entry_grid, exit_grid))
        n_configs = len(configs)
        logger.info("PCA grid search: %d configurations", n_configs)
        t0 = time.perf_counter()

        n_jobs = self.config["experiments"]["n_jobs"]
        results: list[ExperimentResult] = Parallel(n_jobs=n_jobs, verbose=int(verbose))(
            delayed(_run_single_pca)(k, e, x, data, self.config)
            for k, e, x in configs
        )

        elapsed = time.perf_counter() - t0
        df = pd.DataFrame([r.__dict__ for r in results])
        df = df.sort_values("sharpe_with_costs", ascending=False).reset_index(drop=True)

        if verbose:
            best = df.iloc[0]
            logger.info(
                "PCA grid search complete in %.1fs. %d configs tested. "
                "Best Sharpe (with costs) = %.4f at k=%d, entry=%.2f, exit=%.2f",
                elapsed,
                n_configs,
                best["sharpe_with_costs"],
                best["n_factors"],
                best["zscore_entry"],
                best["zscore_exit"],
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
            Dict with keys: n_factors, zscore_entry, zscore_exit.
        """
        best = results_df.loc[results_df[metric].idxmax()]
        return {
            "n_factors": int(best["n_factors"]),
            "zscore_entry": float(best["zscore_entry"]),
            "zscore_exit": float(best["zscore_exit"]),
        }
