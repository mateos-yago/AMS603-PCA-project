"""Yahoo Finance price downloader with batched downloading and Parquet caching."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100        # tickers per yfinance call (avoids silent failures on large batches)
_RETRY_DELAY = 2.0       # seconds between retries
_MAX_RETRIES = 3


class PriceDownloader:
    """Download and cache adjusted-close price data from Yahoo Finance.

    Downloads in batches of ``_BATCH_SIZE`` tickers to avoid silent failures
    that occur when requesting 400+ tickers in a single call. Results are
    cached as Parquet after the first successful download.

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
        """Load prices from Parquet cache if available and non-empty.

        Args:
            start: Start date string.
            end: End date string.

        Returns:
            Price DataFrame or ``None`` if no valid cache exists.
        """
        if start is None:
            start = self.config["data"]["start_date"]
        if end is None:
            end = self.config["data"]["end_date"]

        path = self._cache_path(start, end)
        if path.exists():
            df = pd.read_parquet(path)
            if df.shape[1] == 0:
                logger.warning(
                    "Cached price file %s has 0 columns — treating as invalid, re-downloading.", path
                )
                path.unlink()
                return None
            logger.info("Loading prices from cache: %s (%d tickers × %d days)", path, df.shape[1], df.shape[0])
            return df
        return None

    def _download_batch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Download a single batch and extract the close-price columns.

        Args:
            tickers: Subset of tickers (≤ _BATCH_SIZE).
            start: Start date string.
            end: End date string.

        Returns:
            Close-price DataFrame ``(T, len(tickers))``.
        """
        for attempt in range(_MAX_RETRIES):
            try:
                raw = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                break
            except Exception as exc:
                logger.warning("Download attempt %d/%d failed: %s", attempt + 1, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
                else:
                    raise

        if raw.empty:
            logger.warning("yfinance returned empty DataFrame for batch: %s", tickers[:5])
            return pd.DataFrame()

        # yfinance ≥ 0.2 always returns MultiIndex (price_field, ticker) for multi-ticker calls.
        # With auto_adjust=True the field is "Close", not "Adj Close".
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0).unique().tolist()
            # Prefer configured field; fall back to "Close" when auto_adjust renames it
            field = self.price_field if self.price_field in level0 else "Close"
            if field not in level0:
                logger.error("Neither '%s' nor 'Close' found in columns: %s", self.price_field, level0)
                return pd.DataFrame()
            prices = raw[field].copy()
        else:
            # Single-ticker download returns a plain DataFrame
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else raw.copy()

        return prices

    def download(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Download adjusted close prices from Yahoo Finance in batches.

        Loads from Parquet cache if available. Otherwise downloads in batches
        of ``_BATCH_SIZE`` tickers, concatenates, applies min-history filter,
        forward/backward fills gaps, and caches the result.

        Args:
            tickers: List of ticker symbols in Yahoo Finance format.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            Price DataFrame of shape ``(T, N)`` where T = trading days, N = assets.

        Raises:
            RuntimeError: If the download produces no usable data.
        """
        cache_path = self._cache_path(start, end)
        cached = self.load_cached(start, end)
        if cached is not None:
            return cached

        logger.info(
            "Downloading %d tickers from Yahoo Finance (%s → %s) in batches of %d...",
            len(tickers), start, end, _BATCH_SIZE,
        )

        batches: list[pd.DataFrame] = []
        for i in range(0, len(tickers), _BATCH_SIZE):
            batch = tickers[i: i + _BATCH_SIZE]
            logger.info(
                "  Batch %d/%d (%d tickers)...",
                i // _BATCH_SIZE + 1,
                (len(tickers) + _BATCH_SIZE - 1) // _BATCH_SIZE,
                len(batch),
            )
            df = self._download_batch(batch, start, end)
            if not df.empty:
                batches.append(df)

        if not batches:
            raise RuntimeError(
                "Download produced no data. Check your internet connection and that "
                "the ticker list is valid."
            )

        prices = pd.concat(batches, axis=1)
        # Remove duplicate columns (can arise if a ticker appears in two batches)
        prices = prices.loc[:, ~prices.columns.duplicated()]

        logger.info("Combined download shape before filtering: %s", prices.shape)

        # Drop tickers with insufficient price history
        valid_counts = prices.notna().sum()
        too_short = valid_counts[valid_counts < self.min_history_days].index.tolist()
        if too_short:
            logger.warning(
                "Dropping %d tickers with fewer than %d non-NaN days (showing first 20): %s",
                len(too_short), self.min_history_days, too_short[:20],
            )
            prices = prices.drop(columns=too_short)

        # Forward-fill then backward-fill within each ticker's history
        prices = prices.ffill().bfill()

        # Drop any columns that are still all-NaN after fill
        prices = prices.dropna(axis=1, how="all")

        if prices.shape[1] == 0:
            raise RuntimeError(
                "All tickers were dropped after the min-history filter. "
                f"min_history_days={self.min_history_days}. Check data availability."
            )

        logger.info(
            "Final price DataFrame: %d tickers × %d trading days.", prices.shape[1], prices.shape[0]
        )

        prices.to_parquet(cache_path)
        logger.info("Cached prices to %s", cache_path)
        return prices
