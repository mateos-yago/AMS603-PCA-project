"""Model-agnostic backtesting engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pca_project.backtesting.portfolio import DollarNeutralPortfolio
from pca_project.backtesting.transaction_costs import TransactionCostModel
from pca_project.signals.ou_process import SignalGenerator, ZScoreGenerator

if TYPE_CHECKING:
    from pca_project.factors.base_factor_model import BaseFactorModel

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Core backtesting loop — completely model-agnostic.

    Takes pre-computed signals and raw (non-standardized) returns, simulates
    daily portfolio rebalancing, and records PnL with and without transaction
    costs.

    Signal-to-execution timing: the signal computed at close of day ``t``
    determines the position *entering* day ``t+1``. This is achieved by
    shifting the weight matrix forward by one day before computing PnL.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.portfolio = DollarNeutralPortfolio(config)
        self.tc_model = TransactionCostModel(config)
        self._apply_costs_default: bool = config["backtesting"]["apply_transaction_costs"]

    def run(
        self,
        signals: pd.DataFrame,
        raw_returns: pd.DataFrame,
        apply_costs: bool | None = None,
    ) -> dict:
        """Simulate portfolio performance for a given signal and return series.

        Args:
            signals: Signal DataFrame ``(T, N)`` from SignalGenerator, indexed
                     by decision date.
            raw_returns: Actual (unstandardized) daily log returns ``(T, N)``
                         for the same period.
            apply_costs: If None, uses ``config['backtesting']['apply_transaction_costs']``.

        Returns:
            Dict with keys: daily_returns, cumulative_returns, weights,
            transaction_costs, n_long, n_short, gross_exposure.
        """
        if apply_costs is None:
            apply_costs = self._apply_costs_default

        # Align signal and returns on the intersection of their dates
        common_idx = signals.index.intersection(raw_returns.index)
        signals = signals.loc[common_idx]
        raw_returns = raw_returns.loc[common_idx]

        weights = self.portfolio.compute_weights(signals)         # (T, N)

        # Shift weights forward by 1 day: day-t signal → day-t+1 execution
        weights_shifted = weights.shift(1).fillna(0.0)           # (T, N)

        # Portfolio return = dot(weights_{t-1}, raw_returns_t)
        daily_pnl = (weights_shifted * raw_returns).sum(axis=1)  # (T,)

        # Transaction costs = |Δweight| * cost_rate
        weight_changes = weights_shifted.diff().fillna(weights_shifted)
        tc = self.tc_model.compute_costs(weight_changes)         # (T,)

        if apply_costs:
            daily_returns = daily_pnl - tc
        else:
            daily_returns = daily_pnl

        cumulative_returns = (1.0 + daily_returns).cumprod()

        n_long = (signals == 1).sum(axis=1)
        n_short = (signals == -1).sum(axis=1)
        gross_exposure = weights_shifted.abs().sum(axis=1)

        return {
            "daily_returns": daily_returns,
            "cumulative_returns": cumulative_returns,
            "weights": weights_shifted,
            "transaction_costs": tc,
            "n_long": n_long,
            "n_short": n_short,
            "gross_exposure": gross_exposure,
        }

    def run_with_and_without_costs(
        self,
        signals: pd.DataFrame,
        raw_returns: pd.DataFrame,
    ) -> dict:
        """Run the backtest under both cost scenarios.

        Args:
            signals: Signal DataFrame ``(T, N)``.
            raw_returns: Raw daily log returns ``(T, N)``.

        Returns:
            Dict with keys ``with_costs`` and ``without_costs``, each containing
            the result dict from ``run()``.
        """
        return {
            "with_costs": self.run(signals, raw_returns, apply_costs=True),
            "without_costs": self.run(signals, raw_returns, apply_costs=False),
        }


def run_full_backtest(
    factor_model: "BaseFactorModel",
    raw_returns_test: pd.DataFrame,
    std_returns_test: pd.DataFrame,
    config: dict[str, Any],
    zscore_entry: float | None = None,
    zscore_exit: float | None = None,
) -> dict:
    """End-to-end backtest: residuals → Z-scores → signals → PnL.

    This convenience function orchestrates the entire signal generation and
    backtesting pipeline for a fitted factor model on test-period data.

    Args:
        factor_model: Fitted ``BaseFactorModel`` instance (PCA or AE).
        raw_returns_test: Unstandardized test returns ``(T_test, N)`` — used for PnL.
        std_returns_test: Standardized test returns ``(T_test, N)`` — used for residuals.
        config: Project configuration dict.
        zscore_entry: Z-score entry threshold. Defaults to config value.
        zscore_exit: Z-score exit threshold. Defaults to config value.

    Returns:
        Dict with keys:
          - ``with_costs``: backtest result dict (with transaction costs)
          - ``without_costs``: backtest result dict (without transaction costs)
          - ``signals``: Signal DataFrame
          - ``zscores``: Z-score DataFrame
          - ``ou_params``: OU kappa DataFrame
    """
    logger.info("Running full backtest pipeline...")

    # 1. Get idiosyncratic residuals from the factor model
    residuals = factor_model.get_residuals(std_returns_test)

    # 2. Compute rolling OU Z-scores
    zscore_gen = ZScoreGenerator(config)
    zscores, ou_params = zscore_gen.compute_zscores(residuals)

    # 3. Generate trading signals
    signal_gen = SignalGenerator(config, zscore_entry=zscore_entry, zscore_exit=zscore_exit)
    signals = signal_gen.generate_signals(zscores)

    # 4. Run backtest (with and without costs)
    engine = BacktestEngine(config)
    results = engine.run_with_and_without_costs(signals, raw_returns_test)

    results["signals"] = signals
    results["zscores"] = zscores
    results["ou_params"] = ou_params

    logger.info(
        "Backtest complete. Final cumulative return (with costs): %.4f",
        results["with_costs"]["cumulative_returns"].iloc[-1],
    )
    return results
