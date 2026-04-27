# PCA_project — Full Autonomous Implementation Plan

You are Claude Code. Your task is to implement the entire PCA_project from scratch, following this plan autonomously without waiting for user input between steps.

---

## CRITICAL: Read These First

1. **MASTER INSTRUCTIONS**: `.github/claude/CLAUDE.md` — Read this completely before proceeding.
2. **NUMBERED SUB-INSTRUCTIONS**: `.github/claude/instructions/01_setup.md` through `09_notebook.md` — Follow in order.

---

## Execution Model (Important)

- **main.py** (project root): Runs the entire pipeline end-to-end. Results are exported to `data/results/`.
- **notebooks/analysis.ipynb** (single notebook): Loads pre-computed results and visualizes them. **Never runs experiments or fits models.**
- **config.yaml**: Single source of truth for all parameters.

---

## Autonomous Implementation Sequence

### PHASE 1: Foundation (Instruction 01)

**File**: `.github/claude/instructions/01_setup.md`

**Tasks**:
1. ✅ Create `config.yaml` at project root with the full schema
   - Data parameters (start_date, end_date, universe, cache dirs)
   - Train/val/test splits (70/10/20)
   - PCA grid search ranges
   - Autoencoder grid search ranges (bottleneck, depth, activation)
   - OU process parameters
   - Portfolio leverage and transaction costs
   - All other config shown in instruction 01
2. ✅ Create `requirements.txt` with all dependencies
3. ✅ Create `pca_project/__init__.py` with `load_config()` function
4. ✅ Create `.gitignore` with data/ and other exclusions

**Commit**: `feat: initialize project structure with config and requirements`

---

### PHASE 2: Data Pipeline (Instruction 02)

**File**: `.github/claude/instructions/02_data_pipeline.md`

**Implement in `pca_project/data/`**:

1. ✅ `universe.py` — `SP500Universe` class
   - `get_tickers(as_of_date=None)` — scrape Wikipedia, cache result
   - Log survivorship bias warning every call

2. ✅ `downloader.py` — `PriceDownloader` class
   - `download(tickers, start, end)` — Yahoo Finance via yfinance
   - `load_cached()` — load from Parquet if exists
   - Drop tickers with insufficient history
   - Forward-fill + backward-fill NaN (within-ticker only)
   - Use tqdm for progress

3. ✅ `preprocessor.py` — `DataPreprocessor` class
   - `compute_log_returns(prices)` — compute log returns
   - `cross_sectional_standardize(returns)` — per-timestep standardization
   - `split(returns)` — chronological train/val/test split (no shuffling)
   - `run(prices)` — full pipeline: returns → split → standardize (per-split)
   - Return dict with `train_std`, `test_std`, `train_raw`, etc.
   - Save all to `data/processed/` as Parquet

**Commit**: `feat: implement data pipeline (universe, downloader, preprocessor)`

---

### PHASE 3: Base Factor Model (Instruction 03)

**File**: `.github/claude/instructions/03_pca_model.md`

**Implement in `pca_project/factors/`**:

1. ✅ `base_factor_model.py` — `BaseFactorModel` (ABC)
   - Abstract methods: `fit()`, `get_residuals()`, `get_factor_returns()`
   - Concrete: `validate_not_fitted()` with `_is_fitted` flag

2. ✅ `pca_model.py` — `PCAModel(BaseFactorModel)`
   - `__init__(config, n_factors=None)` — read from config if None
   - `fit(returns)` — Avellaneda & Lee PCA procedure:
     - Correlation matrix from standardized returns
     - Eigendecomposition (top k eigenvectors)
     - Eigenportfolio weights: `Q = v / sigma_i`
     - Factor returns: `F_t = sum(Q * R_t)`
     - Regression to get betas
   - `get_residuals(returns)` — `epsilon = R - (beta * F)`
   - `get_factor_returns(returns)` — reconstructed returns
   - Store: `eigenvalues_`, `eigenvectors_`, `explained_variance_ratio_`, `betas_`, `per_stock_mean_`, `per_stock_std_`
   - `get_variance_explained_summary()` — return dict with eigenvalues, ratios, cumulative

**Commit**: `feat: implement base factor model and PCA model with eigendecomposition`

---

### PHASE 4: Autoencoder (Instruction 04)

**File**: `.github/claude/instructions/04_autoencoder_model.md`

**Implement in `pca_project/factors/`**:

1. ✅ `autoencoder_model.py` — `Autoencoder(nn.Module)`
   - Geometric layer sizing (log-spaced between N and bottleneck)
   - Encoder: linear → activation → dropout → ... → bottleneck (no activation)
   - Decoder: mirrors encoder, final layer has no activation
   - `forward(x)` → `(reconstruction, bottleneck_representation)`
   - `encode(x)` → bottleneck only

2. ✅ `autoencoder_model.py` — `AutoencoderModel(BaseFactorModel)`
   - `__init__(config, bottleneck=None, depth=None, activation=None)`
   - `fit(returns_train, returns_val)` → training loop with early stopping
     - Adam optimizer, MSE loss
     - Track train/val losses
     - Early stopping on val loss
     - Store `network_`, `best_epoch_`, `final_train_loss_`, `final_val_loss_`
   - `get_residuals(returns)` → `actual - reconstruction`
   - `get_factor_returns(returns)` → reconstruction (systematic component)
   - `get_bottleneck_representation(returns)` → latent factors
   - `save(path)`, `load(cls, path, config)` — pickle the whole model

**Commit**: `feat: implement deep autoencoder with PyTorch`

---

### PHASE 5: OU Process & Signals (Instruction 05)

**File**: `.github/claude/instructions/05_ou_signals.md`

**Implement in `pca_project/signals/`**:

1. ✅ `ou_process.py` — `OUProcess` class
   - `estimate_parameters(residuals: np.ndarray)` → dict with a, b, kappa, m, sigma, sigma_eq, is_valid
   - Cumulative residual: `X_k = cumsum(residuals)`
   - AR(1) regression: `X_{n+1} = a + b*X_n + noise`
   - OU params: `kappa = -ln(b)*252`, `m = a/(1-b)`, etc.
   - Validity check: `is_valid = (b < 1) and (kappa >= min_kappa)`

2. ✅ `ou_process.py` — `ZScoreGenerator` class
   - `compute_zscores(residuals: pd.DataFrame)` → (zscores_df, ou_params_df)
   - Rolling window per day (lookback from config)
   - For each stock at each time: estimate OU, compute centred s-score
   - s_score = -m_i / sigma_eq_i + mean(-m_j / sigma_eq_j)
   - Return NaN for invalid OU fits

3. ✅ `ou_process.py` — `SignalGenerator` class
   - `__init__(config, zscore_entry=None, zscore_exit=None)`
   - `generate_signals(zscores)` → position DataFrame with {-1, 0, +1}
   - Stateful position tracking (long/short/flat)
   - Entry: |z| > entry_threshold, Exit: |z| < exit_threshold
   - Handle NaN by closing position

**Commit**: `feat: implement OU process, Z-score, and signal generation`

---

### PHASE 6: Backtesting (Instruction 06)

**File**: `.github/claude/instructions/06_backtesting.md`

**Implement in `pca_project/backtesting/`**:

1. ✅ `transaction_costs.py` — `TransactionCostModel` class
   - Store `cost_bps`, `bid_ask_bps`, `total_one_way_bps`
   - `compute_costs(position_changes, prices=None)` → daily cost as fraction

2. ✅ `portfolio.py` — `DollarNeutralPortfolio` class
   - `compute_weights(signals)` → weight DataFrame
   - Ensure leverage/2 long, leverage/2 short, sum ≈ 0
   - Cap individual weights at max_position_weight

3. ✅ `engine.py` — `BacktestEngine` class
   - `run(signals, raw_returns, apply_costs=None)` → result dict with:
     - daily_returns, cumulative_returns, weights, transaction_costs, n_long, n_short, gross_exposure
   - **Key**: Shift signals by 1 day (day t signal → day t+1 execution)
   - `run_with_and_without_costs()` → runs both scenarios

4. ✅ `engine.py` — `run_full_backtest()` function
   - Takes fitted model, test data, config, zscore thresholds
   - Orchestrates: residuals → Z-scores → signals → backtest
   - Returns combined result dict

**Commit**: `feat: implement backtesting engine with transaction costs`

---

### PHASE 6b: Performance Metrics (Instruction 06b)

**File**: `.github/claude/instructions/06b_metrics.md`

**Implement in `pca_project/metrics/`**:

1. ✅ `performance.py` — `PerformanceAnalyzer` class
   - Store `risk_free_rate_annual` from config
   - Methods:
     - `annualized_return()`, `annualized_volatility()`, `sharpe_ratio()`
     - `maximum_drawdown()`, `max_drawdown_duration()`, `hit_ratio()`, `turnover_rate()`, `calmar_ratio()`
   - `compute_all(daily_returns, weights)` → dict with all metrics
   - `compare(results_a, results_b, label_a, label_b)` → comparison DataFrame

**Commit**: `feat: implement performance metrics analyzer`

---

### PHASE 7: Grid Search & main.py (Instruction 07)

**File**: `.github/claude/instructions/07_experiments.md`

**Implement in `pca_project/experiments/`**:

1. ✅ `__init__.py` — `ExperimentResult` dataclass
   - Fields: model_type, hyperparameters, variance_explained, losses, all metrics (with & without costs)

2. ✅ `__init__.py` — Persistence functions
   - `save_results(results, filename, config)` → pickle to `data/results/`
   - `load_results(filename, config)` → unpickle from disk
   - `results_exist(filename, config)` → bool check

3. ✅ `pca_grid_search.py` — `PCAGridSearch` class
   - `run(data, verbose=True)` → test all n_factors × zscore_entry × zscore_exit combos
   - Use `joblib.Parallel` for parallelization
   - Return DataFrame of results
   - `get_best_config(results_df, metric='sharpe_with_costs')` → best hyperparams

4. ✅ `ae_grid_search.py` — `AEGridSearch` class
   - `run(data, full_grid=False, verbose=True)` → test all AE hyperparameter combos
   - `full_grid=False`: only run default values (fast first pass)
   - `full_grid=True`: full cross-product (warn about compute time)
   - Return DataFrame of results
   - `get_best_config()`

5. ✅ **`main.py`** at **project root** — Master orchestration script
   - Argument parser: `--force`, `--stage {all|data|pca|ae|backtest|metrics}`
   - Stage functions:
     - `stage_data(config, force)` → data dict
     - `stage_pca_grid_search(config, data, force)` → PCA results df
     - `stage_ae_grid_search(config, data, force)` → AE results df
     - `stage_fit_best_models(config, data, pca_results, ae_results, force)` → best models
     - `stage_backtest(config, data, pca_model, ae_model, force)` → backtest results
     - `stage_metrics(config, pca_bt, ae_bt, force)` → computed metrics
   - Each stage checks cache before running (skip if exists, unless --force)
   - Each stage logs: start, cache hit/miss, key stats, duration
   - Export all results to `data/results/`:
     - data_splits.pkl, pca_grid_search.pkl, ae_grid_search.pkl
     - pca_best_model.pkl, ae_best_model.pkl
     - pca_backtest.pkl, ae_backtest.pkl
     - pca_metrics.pkl, ae_metrics.pkl
     - pca_signals.pkl, ae_signals.pkl
     - pca_zscores.pkl, ae_zscores.pkl

**Commit**: `feat: implement grid search and main.py orchestration`

---

### PHASE 8: Visualization (Instruction 08)

**File**: `.github/claude/instructions/08_visualization.md`

**Implement in `pca_project/visualization/`**:

1. ✅ `factor_plots.py` — 5 functions
   - `plot_eigenvalue_spectrum()`, `plot_explained_variance_vs_k()`, `plot_eigenvector_weights()`
   - `plot_autoencoder_loss_curves()`, `plot_reconstruction_quality()`

2. ✅ `signal_plots.py` — 6 functions
   - `plot_zscore_timeseries()`, `plot_residual_acf()`, `plot_ou_parameter_distribution()`
   - `plot_signal_heatmap()`, `plot_position_counts()`, (5 functions total)

3. ✅ `backtest_plots.py` — 5 functions
   - `plot_cumulative_pnl()`, `plot_drawdown()`, `plot_rolling_sharpe()`
   - `plot_monthly_returns_heatmap()`, `plot_transaction_costs_impact()`

4. ✅ `comparison_plots.py` — 6 functions
   - `plot_cumulative_pnl_comparison()`, `plot_metrics_comparison_bar()`, `plot_grid_search_heatmap()`
   - `plot_rolling_sharpe_comparison()`, `plot_correlation_of_returns()`, `create_full_comparison_dashboard()`

All functions:
- Accept `save_path: str | None = None` parameter
- Return the Figure object
- Use consistent colors (PCA = blue, AE = red)
- Include type hints and docstrings

**Commit**: `feat: implement all visualization functions`

---

### PHASE 9: Single Notebook (Instruction 09)

**File**: `.github/claude/instructions/09_notebook.md`

**Create `notebooks/analysis.ipynb`**:

1. ✅ Cell 1: Markdown header with prerequisite notice
2. ✅ Cell 2: Imports, config load, set seeds, setup FIGURES_DIR
3. ✅ Cell 3: Load all 13 results files with FileNotFoundError check if main.py hasn't been run
4. ✅ Section 1: Data overview plots
5. ✅ Section 2: PCA factor analysis
6. ✅ Section 3: Autoencoder analysis
7. ✅ Section 4: Grid search results
8. ✅ Section 5: Signal analysis
9. ✅ Section 6: PCA backtest results
10. ✅ Section 7: Autoencoder backtest results
11. ✅ Section 8: Head-to-head comparison (most important)
12. ✅ Section 9: Summary dashboard + conclusions

**Notebook must NEVER**:
- Import `PCAGridSearch`, `AEGridSearch`, `BacktestEngine`, `DataPreprocessor`, `PCAModel`, `AutoencoderModel`
- Fit any models
- Run any backtests
- Compute residuals or Z-scores
- Recompute anything — load and visualize only

**Commit**: `feat: implement single analysis notebook for visualization`

---

## Final Steps (After All Code)

1. ✅ Create `.github/workflows/` directory with empty `.keep` (for future CI/CD)
2. ✅ Create `data/` directory structure:
   - `data/raw/` — for downloaded prices (gitignored)
   - `data/processed/` — for preprocessed returns (gitignored)
   - `data/results/` — for experiment results (gitignored)
   - `data/results/figures/` — for saved plots
3. ✅ Verify `.gitignore` includes `data/`, `*.pyc`, `.ipynb_checkpoints/`, `.vscode/`, `.idea/`
4. ✅ Create `README.md` (copy from `.github/claude/`)

**Commit**: `chore: create directory structure and finalize setup`

---

## GitHub Commits During Implementation

After EACH phase, push to GitHub with a clean commit message:

```
git add -A
git commit -m "<type>: <message>"
git push origin main
```

Follow the commit message conventions in CLAUDE.md.

---

## Summary: What the User Will Do

1. User runs: `python main.py` (once) — entire pipeline executes, results export to `data/results/`
2. User opens: `notebooks/analysis.ipynb` in Jupyter — loads results, displays all plots
3. That's it. No manual tuning, no re-running, no waiting between steps.

---

## You (Claude Code) Should:

- Read CLAUDE.md completely before starting
- Follow this plan in order, implementing each instruction fully
- Commit to GitHub after each phase
- Ask clarifying questions only if instructions are truly ambiguous
- Never skip steps
- Implement all type hints, docstrings, and error handling
- Test imports locally if possible (e.g., `from pca_project import load_config`)
