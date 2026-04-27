# PCA_project
## Claude Code Master Instructions

---

## Project Overview

This project implements and compares two statistical arbitrage frameworks on S&P 500 equities:

1. **Classical Benchmark**: The PCA-based mean-reversion strategy formalized by Avellaneda & Lee (2010)
2. **ML Challenger**: A Deep Autoencoder that replaces linear PCA with non-linear factor extraction

Both models share the same signal generation pipeline (Ornstein-Uhlenbeck process → Z-score) and the same backtesting engine, making the comparison rigorous and apples-to-apples.

**Key design principles**:
- Strict OOP: every concept is a class, every step is a method
- Maximum modularity: the backtesting engine, data pipeline, and OU signal generator are reusable across both models
- All heavy computation and experiment execution lives in `main.py` — run it once, results are exported to disk
- A single Jupyter notebook (`notebooks/analysis.ipynb`) loads the exported results and produces all visualizations
- A single `config.yaml` governs all tunable parameters

---

## Repository Structure

```
PCA_project/
│
├── .github/
│   └── claude/
│       ├── CLAUDE.md                    ← This file
│       └── instructions/
│           ├── 01_setup.md
│           ├── 02_data_pipeline.md
│           ├── 03_pca_model.md
│           ├── 04_autoencoder_model.md
│           ├── 05_ou_signals.md
│           ├── 06_backtesting.md
│           ├── 06b_metrics.md
│           ├── 07_experiments.md
│           ├── 08_visualization.md
│           └── 09_notebook.md
│
├── pca_project/                         ← Main Python package
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── universe.py                  ← S&P 500 constituent fetching
│   │   ├── downloader.py                ← Yahoo Finance price downloader
│   │   └── preprocessor.py             ← Returns, standardization, train/val/test split
│   │
│   ├── factors/
│   │   ├── __init__.py
│   │   ├── base_factor_model.py         ← Abstract base class for all factor models
│   │   ├── pca_model.py                 ← PCA eigenportfolio extraction & residuals
│   │   └── autoencoder_model.py         ← PyTorch autoencoder & residual extraction
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   └── ou_process.py               ← OU parameter estimation & Z-score generation
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── portfolio.py                 ← Dollar-neutral long/short portfolio construction
│   │   ├── transaction_costs.py         ← Transaction cost & bid-ask spread model
│   │   └── engine.py                    ← Core backtesting loop (model-agnostic)
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── performance.py              ← Sharpe, drawdown, hit ratio, turnover
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── pca_grid_search.py           ← Grid search over k factors & z-score thresholds
│   │   └── ae_grid_search.py            ← Grid search over AE architecture hyperparameters
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── factor_plots.py              ← Eigenvalue spectrum, explained variance plots
│       ├── signal_plots.py              ← Z-score time series, residual diagnostics
│       ├── backtest_plots.py            ← Cumulative PnL, drawdown, rolling Sharpe
│       └── comparison_plots.py          ← Side-by-side PCA vs AE comparison dashboards
│
├── notebooks/
│   └── analysis.ipynb                   ← Single notebook: loads results, produces all plots
│
├── data/
│   ├── raw/                             ← Downloaded price data (gitignored)
│   ├── processed/                       ← Cleaned returns, splits (gitignored)
│   └── results/                         ← All exported experiment results (gitignored)
│
├── main.py                              ← Single entry point: runs entire pipeline end-to-end
├── config.yaml                          ← All tunable parameters (single source of truth)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Execution Model

### Step 1 — Run `main.py`

```bash
python main.py
```

This is the **only** script you ever execute directly. It runs the full pipeline in sequence:

1. Data acquisition and preprocessing
2. PCA model fitting and grid search
3. Autoencoder model training and grid search
4. Backtesting both models (with and without transaction costs)
5. Computing all performance metrics
6. Exporting all results to `data/results/`

Running `main.py` a second time skips any stage whose results already exist on disk (cached), unless `--force` is passed.

### Step 2 — Open `notebooks/analysis.ipynb`

This notebook **never runs experiments**. It only:
- Loads pre-exported results from `data/results/`
- Calls visualization functions from `pca_project/visualization/`
- Displays and saves all plots

If results don't exist yet (i.e., `main.py` hasn't been run), the notebook raises a clear error telling the user to run `main.py` first.

---

## Implementation Instructions

Read the following sub-instruction files in order before writing any code:

1. `./instructions/01_setup.md` — environment, dependencies, config schema
2. `./instructions/02_data_pipeline.md` — data layer classes
3. `./instructions/03_pca_model.md` — PCA factor model
4. `./instructions/04_autoencoder_model.md` — PyTorch autoencoder
5. `./instructions/05_ou_signals.md` — OU process and signal generation
6. `./instructions/06_backtesting.md` — backtesting engine
7. `./instructions/06b_metrics.md` — performance metrics
8. `./instructions/07_experiments.md` — grid search, main.py, and result export
9. `./instructions/08_visualization.md` — all plotting classes
10. `./instructions/09_notebook.md` — notebook structure and content

---

## Non-Negotiable Constraints

- **No data leakage**: The test set must never be seen during training or hyperparameter selection. All OU parameter estimation uses only a rolling lookback window of in-sample data.
- **Chronological splits only**: Never shuffle time series data.
- **Config-driven**: Every number that could change (z-score threshold, lookback window, train split ratio, learning rate, etc.) must live in `config.yaml`. No magic numbers in code.
- **Reproducibility**: Set random seeds (numpy, torch) from config at the top of `main.py` and at the top of the notebook.
- **Type hints everywhere**: All function signatures must include Python type hints.
- **Docstrings**: Every class and public method must have a docstring explaining purpose, args, and returns.
- **Notebook is read-only with respect to computation**: The notebook must never recompute residuals, refit models, or re-run backtests. It only loads and visualizes.

---

## GitHub Workflow

This project uses GitHub for version control. Always follow these rules.

### Always push to GitHub

After every meaningful change (new feature, bug fix, refactor, new analysis stage), commit and push to the `main` branch on GitHub.

- Remote: `git@github.com:mateos-yago/AMS603-PCA-project.git`

### Commit message conventions

- Use the imperative mood: "Add PCA analysis" not "Added PCA analysis"
- First line: short summary (≤72 chars), no period at the end
- If more context is needed, leave a blank line then add bullet points
- Format:
  ```
  <type>: <short summary>

  - Optional detail bullet
  - Another detail
  ```
- Types: `feat` (new feature/analysis), `fix` (bug fix), `refactor`, `data` (data processing step), `docs`, `chore`

### Example commit messages

```
feat: implement PCA on covariance matrix

- Compute eigenvalues and eigenvectors manually
- Plot explained variance ratio
```
```
fix: correct normalization in preprocessing step
```

### Before every commit

1. Run `git status` to see what changed
2. Stage only relevant files (avoid large data files, IDE configs)
3. Write a clean commit message following the conventions above
4. Push immediately: `git push origin main`

### What NOT to commit

- Large data files (`.csv`, `.npy`, `.pkl`, `.parquet`, etc.) — add to `.gitignore`
- IDE configuration (`.idea/`, `.vscode/`)
- Checkpoint files (`.ipynb_checkpoints/`)
- Any secrets or API keys
