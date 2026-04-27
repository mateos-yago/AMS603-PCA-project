# Statistical Arbitrage: PCA vs Deep Autoencoder

A rigorous, modular Python implementation comparing the classical Avellaneda & Lee (2010) PCA-based statistical arbitrage strategy against a Deep Autoencoder challenger on S&P 500 equities.

---

## Overview

This project asks a simple but powerful question: can a non-linear Deep Autoencoder extract better latent market factors than linear PCA, and does this translate into superior out-of-sample trading performance?

Both models share a common signal generation pipeline (Ornstein-Uhlenbeck process → Z-score) and backtesting engine, ensuring an apples-to-apples comparison.

---

## Setup

```bash
git clone <repo>
cd stat_arb
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Project

Execute notebooks in order:

```bash
jupyter lab
```

1. `notebooks/01_data_acquisition.ipynb` — Download & preprocess data (~5–10 min, cached after first run)
2. `notebooks/02_pca_model.ipynb` — PCA analysis and grid search
3. `notebooks/03_autoencoder_model.ipynb` — Autoencoder training and grid search
4. `notebooks/04_backtesting_pca.ipynb` — PCA backtest
5. `notebooks/05_backtesting_autoencoder.ipynb` — Autoencoder backtest
6. `notebooks/06_comparison_analysis.ipynb` — Side-by-side comparison

---

## Configuration

All parameters are in `config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.start_date` | 2014-01-01 | Start of data window |
| `data.end_date` | 2024-01-01 | End of data window |
| `split.train` | 0.70 | Training fraction |
| `pca.default_n_factors` | 15 | PCA components |
| `autoencoder.default_bottleneck` | 15 | AE bottleneck size |
| `signals.default_zscore_entry` | 1.25 | Z-score entry threshold |
| `transaction_costs.cost_bps` | 5 | One-way cost in basis points |

---

## Project Structure

```
stat_arb/
├── config.yaml              ← Single source of truth for all parameters
├── stat_arb/                ← Python package
│   ├── data/                ← Universe, downloader, preprocessor
│   ├── factors/             ← PCA and Autoencoder models
│   ├── signals/             ← OU process and Z-score generation
│   ├── backtesting/         ← Portfolio construction and simulation
│   ├── metrics/             ← Performance metrics
│   ├── experiments/         ← Grid search orchestration
│   └── visualization/       ← All plotting functions
└── notebooks/               ← Execution and visualization layer
```

---

## Key Design Decisions

**Shared backtesting engine**: Both models are evaluated using exactly the same `BacktestEngine`, `DollarNeutralPortfolio`, and `TransactionCostModel`. The only difference is the source of residuals.

**No data leakage**: OU parameters are estimated on a rolling 60-day lookback window. Signals generated on day t are applied to day t+1 returns.

**Transaction costs**: 5 bps one-way + 2 bps half-spread = 7 bps total one-way. Results shown with and without costs.

**Survivorship bias**: The universe is current S&P 500 constituents. This introduces mild survivorship bias — a known limitation documented in the code.

---

## Reference

Avellaneda, M. & Lee, J.-H. (2010). *Statistical arbitrage in the US equities market*. Quantitative Finance, 10(7), 761–782.
