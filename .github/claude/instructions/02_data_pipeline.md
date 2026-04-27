# Instruction 02 — Data Pipeline

All data classes live in `stat_arb/data/`. They must be independent of any model logic.

---

## 2.1 `universe.py` — `SP500Universe`

**Purpose**: Retrieve the list of S&P 500 tickers as of a given date.

### Class: `SP500Universe`

```
SP500Universe(config: dict)
```

**Key method**:

```
get_tickers(as_of_date: str | None = None) -> list[str]
```

**Implementation notes**:
- Scrape the S&P 500 constituent table from Wikipedia (`https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`) using `requests` + `BeautifulSoup`.
- Parse the `<table id="constituents">` and extract the `Symbol` column.
- Clean tickers: replace `.` with `-` (e.g., `BRK.B` → `BRK-B`) to match Yahoo Finance format.
- Cache the result as a JSON file at `data/raw/sp500_tickers_{date}.json` to avoid repeated scraping.
- If `as_of_date` is None, return the current constituents.
- Log the number of tickers found.

**Why**: Fetching current constituents (not historical point-in-time) introduces survivorship bias, which is a known limitation. Log a clear warning about this. True point-in-time constituent data requires a commercial data source. The warning must appear every time `get_tickers()` is called.

---

## 2.2 `downloader.py` — `PriceDownloader`

**Purpose**: Download and cache OHLCV price data from Yahoo Finance.

### Class: `PriceDownloader`

```
PriceDownloader(config: dict)
```

**Key methods**:

```
download(tickers: list[str], start: str, end: str) -> pd.DataFrame
load_cached() -> pd.DataFrame | None
```

**Implementation notes**:
- Use `yfinance.download(tickers, start, end, auto_adjust=True)` with `group_by='ticker'`.
- Extract only the `Adj Close` column (or the field specified in `config['data']['price_field']`).
- Result shape: `(T, N)` where T = trading days, N = number of tickers.
- Cache the raw price DataFrame as a Parquet file at `data/raw/prices_{start}_{end}.parquet`.
- On subsequent calls, load from cache if the file exists and the date range matches.
- Drop tickers where total non-NaN observations are below `config['data']['min_history_days']`. Log which tickers were dropped.
- After download, forward-fill then backward-fill NaN values (handle delisted/halted trading). Apply fill **only within** a ticker's history (do not fill NaN at start of series before first trade).
- Log download progress using `tqdm`.

---

## 2.3 `preprocessor.py` — `DataPreprocessor`

**Purpose**: Transform raw prices into the standardized returns and splits needed by all models.

### Class: `DataPreprocessor`

```
DataPreprocessor(config: dict)
```

**Key methods**:

```
compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame
cross_sectional_standardize(returns: pd.DataFrame) -> pd.DataFrame
split(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
run(prices: pd.DataFrame) -> dict[str, pd.DataFrame]
```

**Detailed specifications**:

### `compute_log_returns`
- Input: price DataFrame `(T, N)`
- Output: log-return DataFrame `(T-1, N)` — `log(P_t / P_{t-1})`
- Drop the first row (NaN from differencing).
- Do **not** standardize here; that is a separate step.

### `cross_sectional_standardize`
- Input: log-return DataFrame `(T, N)`
- For each row (time step) t:
  - Compute cross-sectional mean `mu_t` and std `sigma_t` across all N stocks
  - Replace each return: `R_{i,t} ← (R_{i,t} - mu_t) / sigma_t`
- This is the Avellaneda & Lee standardization that prevents high-volatility stocks from dominating latent factors.
- Return the standardized DataFrame with the same shape and index.

### `split`
- Input: the **raw** (non-standardized) log-return DataFrame
- Uses `config['split']` ratios: train=0.70, val=0.10, test=0.20
- Split is **strictly chronological** — no shuffling
- Returns: `(train_df, val_df, test_df)`
- Log the date ranges and shapes of each split.

### `run` (pipeline method)
- Accepts raw price DataFrame
- Calls: `compute_log_returns` → `split` (on raw returns) → `cross_sectional_standardize` (on each split separately — standardize train, val, test independently to prevent look-ahead bias)
- Returns a dict:
  ```python
  {
    "raw_returns": pd.DataFrame,          # (T, N)
    "train_raw": pd.DataFrame,            # unstandardized, for backtesting PnL
    "val_raw": pd.DataFrame,
    "test_raw": pd.DataFrame,
    "train_std": pd.DataFrame,            # standardized, for model training
    "val_std": pd.DataFrame,
    "test_std": pd.DataFrame,
    "split_dates": dict                   # {"train_start":..., "train_end":..., etc.}
  }
  ```
- Save all DataFrames as Parquet files in `data/processed/`.

**Critical note**: Cross-sectional standardization is fit separately on each split (no parameters are shared across splits). This is correct because standardization parameters (mean, std at each time step) are computed from that time step's own cross-section — they don't leak future information. Document this clearly in the docstring.
