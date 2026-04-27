"""S&P 500 universe fetching with Wikipedia scraping and JSON caching."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class SP500Universe:
    """Retrieve the list of current S&P 500 constituent tickers.

    Note: This fetches *current* constituents from Wikipedia, not point-in-time
    historical membership. This introduces survivorship bias — dead/delisted stocks
    are excluded. A warning is logged on every call to ``get_tickers``.

    Args:
        config: Project configuration dict from ``load_config()``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.cache_dir = Path(config["data"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_tickers(self, as_of_date: str | None = None) -> list[str]:
        """Return the list of S&P 500 tickers.

        Args:
            as_of_date: Ignored for Wikipedia scraping (always returns current
                        constituents). Provided for API consistency.

        Returns:
            Sorted list of ticker symbols in Yahoo Finance format (e.g. BRK-B).
        """
        logger.warning(
            "SURVIVORSHIP BIAS WARNING: S&P 500 constituents are fetched from Wikipedia "
            "and reflect the *current* composition, not historical point-in-time membership. "
            "Stocks that were delisted or removed from the index during the study period are "
            "excluded. This biases backtest results upward. True point-in-time data requires "
            "a commercial data provider (e.g., Compustat, FactSet)."
        )

        today = date.today().isoformat()
        cache_file = self.cache_dir / f"sp500_tickers_{today}.json"

        if cache_file.exists():
            logger.info("Loading S&P 500 tickers from cache: %s", cache_file)
            with cache_file.open("r") as fh:
                tickers: list[str] = json.load(fh)
            logger.info("Loaded %d tickers from cache.", len(tickers))
            return tickers

        logger.info("Scraping S&P 500 tickers from Wikipedia...")
        tickers = self._scrape_wikipedia()

        with cache_file.open("w") as fh:
            json.dump(tickers, fh)
        logger.info("Cached %d tickers to %s", len(tickers), cache_file)
        return tickers

    def _scrape_wikipedia(self) -> list[str]:
        """Scrape constituent table from Wikipedia and return cleaned ticker list."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        try:
            response = requests.get(_WIKI_URL, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch S&P 500 tickers from Wikipedia: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            raise RuntimeError(
                "Could not find the 'constituents' table on the Wikipedia S&P 500 page. "
                "The page structure may have changed."
            )

        tickers: list[str] = []
        for row in table.find_all("tr")[1:]:  # skip header
            cells = row.find_all("td")
            if cells:
                raw = cells[0].get_text(strip=True)
                # Yahoo Finance uses '-' instead of '.' (e.g. BRK.B → BRK-B)
                tickers.append(raw.replace(".", "-"))

        tickers = sorted(set(tickers))
        logger.info("Scraped %d unique tickers from Wikipedia.", len(tickers))
        return tickers
