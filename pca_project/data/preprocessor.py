"""Return computation, cross-sectional standardization, and train/val/test splitting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Transform raw prices into standardized returns and chronological splits.

    Pipeline:
      1. Compute log returns from prices.
      2. Split into train / val / test chronologically (no shuffling).
      3. Cross-sectionally standardize each split independently to prevent
         look-ahead bias (each split's mean/std are computed only from that split).

    Cross-sectional standardization (Avellaneda & Lee): at each time step t,
    subtract the cross-sectional mean and divide by the cross-sectional std.
    This prevents high-volatility stocks from dominating the latent factors.
    Standardization parameters are derived per time step from that step's own
    cross-section — no information leaks across time.

    Args:
        config: Project configuration dict from ``load_config()``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._split_ratios = config["split"]

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns from a price DataFrame.

        Args:
            prices: Price DataFrame of shape ``(T, N)``.

        Returns:
            Log-return DataFrame of shape ``(T-1, N)``. First row (NaN) is dropped.
        """
        log_returns = np.log(prices / prices.shift(1)).iloc[1:]
        logger.info("Computed log returns: shape %s", log_returns.shape)
        return log_returns

    def cross_sectional_standardize(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Standardize returns cross-sectionally at each time step.

        For each row t: R_{i,t} ← (R_{i,t} − μ_t) / σ_t
        where μ_t and σ_t are the cross-sectional mean and std across all N stocks.

        Args:
            returns: Log-return DataFrame of shape ``(T, N)``.

        Returns:
            Standardized DataFrame with the same shape and index.
        """
        mu = returns.mean(axis=1)
        sigma = returns.std(axis=1, ddof=1)
        # Broadcast: subtract row mean and divide by row std
        standardized = returns.sub(mu, axis=0).div(sigma, axis=0)
        return standardized

    def split(
        self, returns: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronologically split returns into train, validation, and test sets.

        Split ratios come from ``config['split']``. No shuffling is performed.

        Args:
            returns: Log-return DataFrame of shape ``(T, N)``.

        Returns:
            Tuple of ``(train_df, val_df, test_df)``.
        """
        T = len(returns)
        train_end = int(T * self._split_ratios["train"])
        val_end = train_end + int(T * self._split_ratios["val"])

        train = returns.iloc[:train_end]
        val = returns.iloc[train_end:val_end]
        test = returns.iloc[val_end:]

        for name, split in [("train", train), ("val", val), ("test", test)]:
            logger.info(
                "Split %-5s: %s → %s  (%d days, %d assets)",
                name,
                split.index[0].date() if len(split) else "N/A",
                split.index[-1].date() if len(split) else "N/A",
                len(split),
                split.shape[1],
            )
        return train, val, test

    def run(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
        """Full preprocessing pipeline: prices → split standardized returns.

        Saves all DataFrames as Parquet files in ``data/processed/``.

        Args:
            prices: Raw price DataFrame of shape ``(T, N)``.

        Returns:
            Dict with keys:
              - ``raw_returns``  : full log-return DataFrame (T, N)
              - ``train_raw``    : unstandardized train returns (for PnL computation)
              - ``val_raw``      : unstandardized val returns
              - ``test_raw``     : unstandardized test returns
              - ``train_std``    : standardized train returns (for model fitting)
              - ``val_std``      : standardized val returns (for early stopping)
              - ``test_std``     : standardized test returns (for residual/backtest)
              - ``split_dates``  : dict of date boundaries
        """
        raw_returns = self.compute_log_returns(prices)
        train_raw, val_raw, test_raw = self.split(raw_returns)

        train_std = self.cross_sectional_standardize(train_raw)
        val_std = self.cross_sectional_standardize(val_raw)
        test_std = self.cross_sectional_standardize(test_raw)

        split_dates = {
            "train_start": str(train_raw.index[0].date()),
            "train_end": str(train_raw.index[-1].date()),
            "val_start": str(val_raw.index[0].date()),
            "val_end": str(val_raw.index[-1].date()),
            "test_start": str(test_raw.index[0].date()),
            "test_end": str(test_raw.index[-1].date()),
        }

        data = {
            "raw_returns": raw_returns,
            "train_raw": train_raw,
            "val_raw": val_raw,
            "test_raw": test_raw,
            "train_std": train_std,
            "val_std": val_std,
            "test_std": test_std,
            "split_dates": split_dates,
        }

        # Persist to disk
        for key, df in data.items():
            if isinstance(df, pd.DataFrame):
                out_path = self.processed_dir / f"{key}.parquet"
                df.to_parquet(out_path)
                logger.debug("Saved %s to %s", key, out_path)

        logger.info("Preprocessing complete. Split dates: %s", split_dates)
        return data
