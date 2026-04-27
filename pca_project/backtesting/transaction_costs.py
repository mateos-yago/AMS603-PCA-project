"""Transaction cost model for portfolio simulation."""

from __future__ import annotations

from typing import Any

import pandas as pd


class TransactionCostModel:
    """Simple proportional transaction cost model.

    Total one-way cost = slippage + half bid-ask spread, expressed in basis
    points and applied to the absolute change in portfolio weight each day.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        tc_cfg = config["transaction_costs"]
        self.cost_bps: float = tc_cfg["cost_bps"]
        self.bid_ask_bps: float = tc_cfg["bid_ask_spread_bps"]
        self.total_one_way_bps: float = self.cost_bps + self.bid_ask_bps

    def compute_costs(
        self,
        position_changes: pd.DataFrame,
        prices: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Compute daily transaction costs as a fraction of portfolio value.

        Args:
            position_changes: Absolute change in weight per stock per day ``(T, N)``.
            prices: Optional price DataFrame (unused in weight-space model).

        Returns:
            Series ``(T,)`` — daily cost as a fraction of portfolio value.
        """
        cost_fraction = self.total_one_way_bps / 10_000
        daily_costs = position_changes.abs().sum(axis=1) * cost_fraction
        return daily_costs
