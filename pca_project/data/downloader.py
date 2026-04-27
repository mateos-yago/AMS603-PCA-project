"""Yahoo Finance price downloader with Parquet caching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceDownloader:
    """Download and cache adjusted-close price data from Yahoo Finance.

    Downloads data once, caches as Parquet, and loads from cache on subsequent
    runs. Tickers with insufficient price history are dropped with a log warning.

    Args:
        config: Project configuration dict from ``load_config()``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.cache_dir = Path(config["data"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.price_field = config["data"]["price_field"]
        self.min_history_days = config["data"]["min_history_days"]

    def _cache_path(self, start: str, end: str) -> Path:
        return self.cache_dir / f"prices_{start}_{end}.parquet"

    def load_cached(self, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
        """Load prices from Parquet cache if available.

        Args:
            start: Start date string (used to locate the cache file).
            end: End date string.

        Returns:
            Price DataFrame or ``None`` if no cache exists.
        """
        if start is None:
            start = self.config["data"]["start_date"]
        if end is None:
            end = self.config["data"]["end_date"]

        path = self._cache_path(start, end)
        if path.exists():
            logger.info("Loading prices from cache: %s", path)
            return pd.read_parquet(path)
        return None

    def download(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Download adjusted close prices from Yahoo Finance.

        Loads from Parquet cache if available. Otherwise downloads fresh data,
        drops tickers with insufficient history, forward/backward fills NaNs,
        and caches the result.

        Args:
            tickers: List of ticker symbols in Yahoo Finance format.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            Price DataFrame of shape ``(T, N)`` where T = trading days, N = assets.
        """
        cache_path = self._cache_path(start, end)
        cached = self.load_cached(start, end)
        if cached is not None:
            logger.info(
                "Loaded %d tickers × %d days from cache.", cached.shape[1], cached.shape[0]
            )
            return cached

        logger.info("Downloading %d tickers from Yahoo Finance (%s → %s)...", len(tickers), start, end)

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=True,
        )

        # yfinance returns a MultiIndex when multiple tickers are requested.
        # The top level is the price field; second level is the ticker symbol.
        if isinstance(raw.columns, pd.MultiIndex):
            if self.price_field in raw.columns.get_level_values(0):
                prices = raw[self.price_field].copy()
            else:
                # auto_adjust=True replaces "Adj Close" with "Close"
                prices = raw["Close"].copy()
        else:
            prices = raw.copy()

        logger.info("Raw download shape: %s", prices.shape)

        # Drop tickers with insufficient price history
        valid_counts = prices.notna().sum()
        too_short = valid_counts[valid_counts < self.min_history_days].index.tolist()
        if too_short:
            logger.warning(
                "Dropping %d tickers with fewer than %d non-NaN days: %s",
                len(too_short),
                self.min_history_days,
                too_short[:20],  # truncate log line
            )
            prices = prices.drop(columns=too_short)

        # Forward-fill then backward-fill within each ticker's history
        # This handles halted trading days and brief data gaps.
        prices = prices.ffill().bfill()

        # After fill, drop any remaining all-NaN columns
        prices = prices.dropna(axis=1, how="all")

        logger.info(
            "Final price DataFrame: %d tickers × %d trading days.", prices.shape[1], prices.shape[0]
        )

        prices.to_parquet(cache_path)
        logger.info("Cached prices to %s", cache_path)
        return prices
