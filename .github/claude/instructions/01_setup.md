# Instruction 01 — Setup, Dependencies & Configuration

## 1.1 Python Environment

- Python version: **3.11**
- Create a virtual environment: `python -m venv .venv`
- Install dependencies from `requirements.txt`

## 1.2 requirements.txt

Generate a `requirements.txt` with at minimum the following packages (use latest stable versions unless noted):

```
numpy
pandas
scipy
scikit-learn
torch                  # CPU build is fine; GPU optional
yfinance
requests
beautifulsoup4         # for S&P 500 constituent scraping
matplotlib
seaborn
plotly
jupyter
ipykernel
pyyaml
tqdm
statsmodels            # for AR(1) regression in OU estimation
joblib                 # for parallel grid search
```

## 1.3 config.yaml — Complete Schema

Create `config.yaml` at the project root. Every parameter that could conceivably change must live here. The file must be structured in clearly named sections.

```yaml
# ============================================================
# GLOBAL
# ============================================================
random_seed: 42
log_level: "INFO"

# ============================================================
# DATA
# ============================================================
data:
  start_date: "2014-01-01"        # 10-year window ending ~today
  end_date: "2024-01-01"
  universe: "sp500"               # only supported value for now
  price_field: "Adj Close"
  min_history_days: 504           # drop tickers with fewer days (~2 years)
  cache_dir: "data/raw"
  processed_dir: "data/processed"
  results_dir: "data/results"

# ============================================================
# TRAIN / VAL / TEST SPLIT  (must sum to 1.0)
# ============================================================
split:
  train: 0.70
  val:   0.10
  test:  0.20

# ============================================================
# PCA MODEL
# ============================================================
pca:
  # Grid search values — lists; first value is the default run
  n_factors_grid: [5, 10, 15, 20, 30]
  default_n_factors: 15
  correlation_window: 252         # days used to compute correlation matrix (Avellaneda & Lee)
  variance_threshold: 0.55        # alternative: select k to explain this fraction of variance

# ============================================================
# AUTOENCODER MODEL
# ============================================================
autoencoder:
  # Bottleneck sizes to test (should mirror pca.n_factors_grid for fair comparison)
  bottleneck_grid: [5, 10, 15, 20, 30]
  default_bottleneck: 15

  # Hidden layer depth options
  depth_grid: [1, 2, 3]           # number of hidden layers on each side of bottleneck
  default_depth: 2

  # Activation functions to test
  activation_grid: ["tanh", "elu", "relu"]
  default_activation: "tanh"

  # Training hyperparameters
  learning_rate: 0.001
  weight_decay: 0.0001            # L2 regularization
  batch_size: 64
  max_epochs: 200
  early_stopping_patience: 15
  dropout_rate: 0.1

  # Architecture scaling factor: hidden layer sizes = bottleneck * scale^layer_index
  hidden_scale: 4                 # e.g. bottleneck=15, depth=2 → layers: [N, 240, 60, 15, 60, 240, N]

# ============================================================
# OU PROCESS & SIGNAL GENERATION
# ============================================================
signals:
  ou_lookback_days: 60            # rolling window for OU parameter estimation (Avellaneda & Lee)
  min_kappa: 8.4                  # minimum speed of mean reversion (= 252/30); reject slower stocks
  
  # Z-score thresholds — grid searched
  zscore_entry_grid: [1.0, 1.25, 1.5, 2.0]
  zscore_exit_grid:  [0.0, 0.5,  0.75]
  default_zscore_entry: 1.25
  default_zscore_exit:  0.0       # simplified: close when Z crosses zero

# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================
portfolio:
  leverage: 2.0                   # dollar amount long (and short) per unit of equity
  max_position_weight: 0.05       # max fraction of equity in any single stock

# ============================================================
# TRANSACTION COSTS
# ============================================================
transaction_costs:
  cost_bps: 5                     # one-way slippage in basis points (5 bps = 0.05%)
  bid_ask_spread_bps: 2           # half-spread in basis points

# ============================================================
# BACKTESTING
# ============================================================
backtesting:
  initial_capital: 1_000_000     # USD
  rebalance_frequency: "daily"
  apply_transaction_costs: true   # toggle for cost vs no-cost comparison
  risk_free_rate_annual: 0.04     # for Sharpe ratio calculation

# ============================================================
# EXPERIMENTS / GRID SEARCH
# ============================================================
experiments:
  n_jobs: -1                      # parallel jobs for grid search (-1 = all cores)
  save_all_results: true
  results_filename: "grid_search_results.csv"
```

## 1.4 Config Loading Utility

In `stat_arb/__init__.py`, implement a `load_config()` function that:
- Reads `config.yaml` using PyYAML
- Returns the config as a plain Python dict
- Accepts an optional `path` argument (defaults to project root `config.yaml`)
- Is importable as `from stat_arb import load_config`

All classes and scripts must call `load_config()` rather than hard-coding values.
