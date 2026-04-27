"""Portfolio performance metrics analyzer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """Compute standard portfolio performance metrics from daily return series.

    All metric methods accept a ``pd.Series`` of simple (not log) daily returns.

    Args:
        config: Project configuration dict.
    """

    TRADING_DAYS = 252

    def __init__(self, config: dict[str, Any]) -> None:
        self.rf_annual: float = config["backtesting"]["risk_free_rate_annual"]
        self.rf_daily: float = (1.0 + self.rf_annual) ** (1.0 / self.TRADING_DAYS) - 1.0

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def annualized_return(self, daily_returns: pd.Series) -> float:
        """Annualized geometric mean return.

        Args:
            daily_returns: Series of daily simple returns.

        Returns:
            Annualized return as a decimal.
        """
        mean_daily = daily_returns.mean()
        return (1.0 + mean_daily) ** self.TRADING_DAYS - 1.0

    def annualized_volatility(self, daily_returns: pd.Series) -> float:
        """Annualized standard deviation of daily returns.

        Args:
            daily_returns: Series of daily simple returns.

        Returns:
            Annualized volatility as a decimal.
        """
        return float(daily_returns.std(ddof=1) * np.sqrt(self.TRADING_DAYS))

    def sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """Annualized Sharpe ratio (excess return over risk-free rate).

        Args:
            daily_returns: Series of daily simple returns.

        Returns:
            Sharpe ratio.
        """
        excess = daily_returns - self.rf_daily
        if excess.std(ddof=1) == 0:
            return 0.0
        return float(excess.mean() / excess.std(ddof=1) * np.sqrt(self.TRADING_DAYS))

    def maximum_drawdown(self, daily_returns: pd.Series) -> float:
        """Maximum peak-to-trough drawdown.

        Args:
            daily_returns: Series of daily simple returns.

        Returns:
            Maximum drawdown as a negative decimal.
        """
        wealth = (1.0 + daily_returns).cumprod()
        rolling_max = wealth.cummax()
        drawdown = (wealth - rolling_max) / rolling_max
        return float(drawdown.min())

    def max_drawdown_duration(self, daily_returns: pd.Series) -> int:
        """Length in days of the longest drawdown period.

        Args:
            daily_returns: Series of daily simple returns.

        Returns:
            Number of trading days in the longest drawdown period.
        """
        wealth = (1.0 + daily_returns).cumprod()
        rolling_max = wealth.cummax()
        in_drawdown = wealth < rolling_max

        max_duration = 0
        current_duration = 0
        for flag in in_drawdown:
            if flag:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return max_duration

    def hit_ratio(self, daily_returns: pd.Series) -> float:
        """Fraction of trading days with positive returns.

        Args:
            daily_returns: Series of daily simple returns.

        Returns:
            Hit ratio in [0, 1].
        """
        if len(daily_returns) == 0:
            return 0.0
        return float((daily_returns > 0).sum() / len(daily_returns))

    def turnover_rate(self, weights: pd.DataFrame) -> float:
        """Average annualized one-way portfolio turnover.

        Args:
            weights: Portfolio weight DataFrame ``(T, N)``.

        Returns:
            Annualized turnover rate (daily mean × 252).
        """
        daily_turnover = weights.diff().abs().sum(axis=1).mean()
        return float(daily_turnover * self.TRADING_DAYS)

    def calmar_ratio(self, daily_returns: pd.Series) -> float:
        """Calmar ratio: annualized return / |maximum drawdown|.

        Args:
            daily_returns: Series of daily simple returns.

        Returns:
            Calmar ratio (or NaN if MDD is zero).
        """
        mdd = self.maximum_drawdown(daily_returns)
        ann_ret = self.annualized_return(daily_returns)
        if mdd == 0.0:
            return np.nan
        return ann_ret / abs(mdd)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def compute_all(
        self, daily_returns: pd.Series, weights: pd.DataFrame
    ) -> dict[str, float | int]:
        """Compute all performance metrics in one call.

        Args:
            daily_returns: Series of daily simple returns.
            weights: Portfolio weight DataFrame ``(T, N)``.

        Returns:
            Dict containing all metrics.
        """
        daily_to = weights.diff().abs().sum(axis=1).mean()
        total_return = float((1.0 + daily_returns).prod() - 1.0)

        return {
            "annualized_return": self.annualized_return(daily_returns),
            "annualized_volatility": self.annualized_volatility(daily_returns),
            "sharpe_ratio": self.sharpe_ratio(daily_returns),
            "maximum_drawdown": self.maximum_drawdown(daily_returns),
            "max_drawdown_duration_days": self.max_drawdown_duration(daily_returns),
            "hit_ratio": self.hit_ratio(daily_returns),
            "daily_turnover": float(daily_to),
            "annualized_turnover": float(daily_to * self.TRADING_DAYS),
            "calmar_ratio": self.calmar_ratio(daily_returns),
            "total_return": total_return,
            "n_trading_days": len(daily_returns),
        }

    def compare(
        self,
        results_a: dict,
        results_b: dict,
        label_a: str,
        label_b: str,
    ) -> pd.DataFrame:
        """Side-by-side comparison table of two result sets.

        Args:
            results_a: Metrics dict for model A (from ``compute_all``).
            results_b: Metrics dict for model B.
            label_a: Column label for model A.
            label_b: Column label for model B.

        Returns:
            DataFrame with one row per metric, two columns.
        """
        df = pd.DataFrame({label_a: results_a, label_b: results_b})
        return df
