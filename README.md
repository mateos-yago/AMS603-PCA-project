# Statistical Arbitrage: PCA vs Deep Autoencoder

A rigorous, modular Python implementation comparing the classical Avellaneda & Lee (2010) PCA-based statistical arbitrage strategy against a Deep Autoencoder challenger on S&P 500 equities.

---

## Overview

This project asks a simple but powerful question: can a non-linear Deep Autoencoder extract better latent market factors than linear PCA, and does this translate into superior out-of-sample trading performance?

Both models share a common signal generation pipeline (Ornstein-Uhlenbeck process → Z-score) and backtesting engine, ensuring an apples-to-apples comparison.

---

## Setup

```bash
git clone git@github.com:mateos-yago/AMS603-PCA-project.git
cd AMS603-PCA-project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Run the full pipeline

```bash
python main.py
```

This single command executes the entire pipeline:
1. Downloads S&P 500 price data from Yahoo Finance (cached after first run)
2. Runs PCA grid search over `n_factors × zscore_entry × zscore_exit`
3. Runs Autoencoder grid search over `depth × activation`
4. Fits the best model for each approach
5. Backtests both models with and without transaction costs
6. Computes all performance metrics
7. Exports all results to `data/results/`

**Options**:
```bash
python main.py --force           # re-run all stages, overwrite cached results
python main.py --stage data      # run only the data download stage
python main.py --stage pca       # run only the PCA grid search
python main.py --stage ae        # run only the AE grid search
python main.py --stage metrics   # run only the metrics computation
```

### Step 2 — Open the notebook

```bash
jupyter lab notebooks/analysis.ipynb
```

The notebook loads all pre-computed results from `data/results/` and produces every visualization. **Never re-runs experiments** — it only loads and plots.

---

## Configuration

All parameters live in `config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.start_date` | 2014-01-01 | Start of data window |
| `data.end_date` | 2024-01-01 | End of data window |
| `split.train` | 0.70 | Training fraction |
| `pca.default_n_factors` | 15 | PCA components |
| `pca.n_factors_grid` | [5,10,15,20,30] | Grid search values |
| `autoencoder.default_bottleneck` | 15 | AE bottleneck size |
| `autoencoder.depth_grid` | [1,2,3] | Hidden layers per side |
| `signals.default_zscore_entry` | 1.25 | Z-score entry threshold |
| `signals.ou_lookback_days` | 60 | OU estimation window |
| `signals.min_kappa` | 8.4 | Min mean-reversion speed |
| `transaction_costs.cost_bps` | 5 | One-way slippage (bps) |
| `backtesting.initial_capital` | 1,000,000 | Starting capital (USD) |

---

## Project Structure

```
PCA_project/
├── main.py                      ← Single entry point: runs full pipeline
├── config.yaml                  ← All tunable parameters
├── requirements.txt
│
├── pca_project/                 ← Python package
│   ├── data/                    ← SP500Universe, PriceDownloader, DataPreprocessor
│   ├── factors/                 ← BaseFactorModel, PCAModel, AutoencoderModel
│   ├── signals/                 ← OUProcess, ZScoreGenerator, SignalGenerator
│   ├── backtesting/             ← BacktestEngine, DollarNeutralPortfolio, TransactionCostModel
│   ├── metrics/                 ← PerformanceAnalyzer
│   ├── experiments/             ← PCAGridSearch, AEGridSearch, ExperimentResult
│   └── visualization/           ← factor_plots, signal_plots, backtest_plots, comparison_plots
│
├── notebooks/
│   └── analysis.ipynb           ← Visualization only — never runs experiments
│
└── data/
    ├── raw/                     ← Downloaded prices (gitignored)
    ├── processed/               ← Preprocessed returns (gitignored)
    └── results/                 ← All experiment results (gitignored)
```

---

## Key Design Decisions

**Single entry point**: `python main.py` is the only script you ever execute. The notebook is read-only with respect to computation.

**Cached stages**: Every stage checks for existing results before running. Re-run with `--force` to overwrite.

**No data leakage**: OU parameters are estimated on a rolling 60-day lookback window. Signals generated on day t are applied to day t+1 returns. Test data is never seen during training or validation.

**Shared backtesting engine**: Both models are evaluated using the same `BacktestEngine`, `DollarNeutralPortfolio`, and `TransactionCostModel`. The only difference is the source of residuals.

**Transaction costs**: 5 bps one-way slippage + 2 bps half-spread = 7 bps total one-way. Results shown with and without costs.

**Survivorship bias**: The universe is current S&P 500 constituents. This introduces mild upward bias — documented and warned in the code.

---

## Reference

Avellaneda, M. & Lee, J.-H. (2010). *Statistical arbitrage in the US equities market*. Quantitative Finance, 10(7), 761–782.
