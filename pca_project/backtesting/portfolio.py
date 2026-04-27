"""Dollar-neutral long/short portfolio weight computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DollarNeutralPortfolio:
    """Convert raw signals into dollar-neutral portfolio weights.

    Each day, the portfolio is split evenly across all long positions (total
    weight = leverage/2) and all short positions (total weight = -leverage/2),
    yielding a net exposure of approximately zero.

    Individual weights are capped at ``max_position_weight``; excess is
    redistributed pro-rata to other positions on the same side.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        port_cfg = config["portfolio"]
        self.leverage: float = port_cfg["leverage"]
        self.max_weight: float = port_cfg["max_position_weight"]

    def compute_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signal DataFrame into dollar-neutral weight DataFrame.

        Args:
            signals: Signal DataFrame ``(T, N)`` with values in ``{-1, 0, +1}``.

        Returns:
            Weight DataFrame ``(T, N)`` summing to approximately 0 each row.
        """
        T, N = signals.shape
        weights_np = np.zeros((T, N))

        half_lev = self.leverage / 2.0

        for t in range(T):
            s_t = signals.values[t]
            long_mask = s_t == 1
            short_mask = s_t == -1

            n_long = long_mask.sum()
            n_short = short_mask.sum()

            if n_long == 0 or n_short == 0:
                # Cannot form a dollar-neutral portfolio without both sides
                continue

            raw_long_w = half_lev / n_long
            raw_short_w = -half_lev / n_short

            w_t = np.zeros(N)
            w_t[long_mask] = raw_long_w
            w_t[short_mask] = raw_short_w

            # Cap individual weights at max_position_weight
            w_t = self._apply_weight_cap(w_t, long_mask, short_mask)
            weights_np[t] = w_t

        return pd.DataFrame(weights_np, index=signals.index, columns=signals.columns)

    def _apply_weight_cap(
        self,
        w: np.ndarray,
        long_mask: np.ndarray,
        short_mask: np.ndarray,
    ) -> np.ndarray:
        """Cap weights at max_position_weight and redistribute excess pro-rata.

        Args:
            w: Weight array for a single day.
            long_mask: Boolean mask for long positions.
            short_mask: Boolean mask for short positions.

        Returns:
            Capped and renormalized weight array.
        """
        cap = self.max_weight
        half_lev = self.leverage / 2.0

        for mask in (long_mask, short_mask):
            if mask.sum() == 0:
                continue
            # Absolute weights for this side
            w_side = np.abs(w[mask])
            sign = np.sign(w[mask][0])

            # Iteratively cap and redistribute
            for _ in range(20):  # converges in a few iterations
                over = w_side > cap
                if not over.any():
                    break
                excess = (w_side[over] - cap).sum()
                w_side[over] = cap
                n_under = (~over).sum()
                if n_under > 0:
                    w_side[~over] += excess / n_under

            # Renormalize so the side's total equals half_lev
            total = w_side.sum()
            if total > 0:
                w_side = w_side * (half_lev / total)

            w[mask] = sign * w_side

        return w
