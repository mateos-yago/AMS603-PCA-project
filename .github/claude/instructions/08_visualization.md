# Instruction 08 — Visualization

All plotting logic lives in `stat_arb/visualization/`. Every plot function must:
- Accept a `save_path: str | None = None` parameter — if provided, save the figure to that path
- Return the `matplotlib.figure.Figure` object
- Use a consistent style (set once at module level: `plt.style.use('seaborn-v0_8-whitegrid')`)
- Use a consistent color palette: PCA = `#2E86AB` (blue), Autoencoder = `#E84855` (red)
- Label all axes, set informative titles, include legends where applicable

---

## 8.1 `factor_plots.py` — Factor Model Diagnostics

### `plot_eigenvalue_spectrum(eigenvalues, explained_ratios, n_factors_selected, save_path=None)`
- Bar chart of top 50 eigenvalues (percentage of variance explained)
- Highlight the selected top `n_factors_selected` bars in the PCA color
- Add a vertical dashed line at the cutoff
- Add cumulative variance line on secondary y-axis
- Title: "Eigenvalue Spectrum — PCA Factor Selection"

### `plot_explained_variance_vs_k(k_values, variance_explained, save_path=None)`
- Line chart: x = number of factors k, y = cumulative variance explained (%)
- Mark specific grid-searched k values
- Draw horizontal dashed lines at 45%, 55%, 65% variance levels
- Title: "Cumulative Explained Variance by Number of PCA Factors"

### `plot_eigenvector_weights(eigenvector, tickers, component_idx, save_path=None)`
- Horizontal bar chart of the eigenvector component weights for a selected PC
- Sort by weight magnitude
- Color positive weights blue, negative red
- Title: f"Eigenvector {component_idx+1} — Stock Weights"

### `plot_autoencoder_loss_curves(train_losses, val_losses, best_epoch, save_path=None)`
- Line chart of train loss and val loss vs epoch
- Vertical dashed line at `best_epoch`
- Log-scale y-axis optional (add toggle)
- Title: "Autoencoder Training & Validation Loss"

### `plot_reconstruction_quality(actual_returns, reconstructed_returns, ticker, save_path=None)`
- Two-panel figure for a single stock:
  - Top: overlay actual vs reconstructed returns time series
  - Bottom: scatter plot (actual vs reconstructed) with R² annotation
- Title: f"Reconstruction Quality — {ticker}"

---

## 8.2 `signal_plots.py` — Signal & Residual Diagnostics

### `plot_zscore_timeseries(zscores, ticker, entry_threshold, exit_threshold, signals, save_path=None)`
- Line chart of Z-score for a single stock over time
- Horizontal dashed lines at ±entry_threshold (entry) and ±exit_threshold (exit)
- Color background regions: green when long, red when short, grey when flat
- Title: f"Z-Score Time Series — {ticker}"

### `plot_residual_acf(residuals, ticker, n_lags=40, save_path=None)`
- Autocorrelation function bar plot of the residual series for one stock
- Add 95% confidence bands (blue shaded region at ±1.96/sqrt(T))
- Title: f"Residual ACF — {ticker}"

### `plot_ou_parameter_distribution(ou_params_df, param='kappa', save_path=None)`
- Histogram of a chosen OU parameter across all stocks
- Add vertical dashed line at the mean and at `min_kappa` threshold
- Title: f"Distribution of OU Parameter: {param}"

### `plot_signal_heatmap(signals, start_date=None, end_date=None, save_path=None)`
- Heatmap: x = time, y = stocks (subset — top 50 by signal frequency)
- Color: red = short, green = long, white = no position
- Title: "Signal Map — Long/Short Positions Over Time"

### `plot_position_counts(n_long, n_short, save_path=None)`
- Stacked area chart over time:
  - Top half: number of long positions (positive, green)
  - Bottom half: number of short positions (negative, red, reflected)
- Title: "Number of Active Long and Short Positions"

---

## 8.3 `backtest_plots.py` — Portfolio Performance

### `plot_cumulative_pnl(daily_returns_with_costs, daily_returns_no_costs, label, save_path=None)`
- Line chart: two lines (with vs without transaction costs)
- X-axis: date, Y-axis: cumulative return (indexed to 1.0 at start)
- Shade the gap between the two lines
- Annotate final cumulative return for both
- Title: f"Cumulative PnL — {label}"

### `plot_drawdown(daily_returns, label, save_path=None)`
- Area chart of rolling drawdown (always ≤ 0)
- Shade in red
- Mark the maximum drawdown point with annotation
- Title: f"Drawdown — {label}"

### `plot_rolling_sharpe(daily_returns, window=63, label, save_path=None)`
- Rolling Sharpe ratio (annualized) with a 63-day (≈ 1 quarter) window
- Horizontal dashed line at 0 and at 1.0
- Title: f"Rolling {window}-Day Sharpe Ratio — {label}"

### `plot_monthly_returns_heatmap(daily_returns, label, save_path=None)`
- Pivot monthly returns into a calendar heatmap (years × months)
- Color: green = positive, red = negative
- Annotate each cell with return percentage
- Title: f"Monthly Returns — {label}"

### `plot_transaction_costs_impact(daily_returns_with_costs, daily_returns_no_costs, transaction_costs, label, save_path=None)`
- Three-panel figure:
  - Panel 1: cumulative PnL with and without costs
  - Panel 2: daily transaction cost as % of portfolio
  - Panel 3: rolling 63-day cumulative cost
- Title: f"Transaction Cost Analysis — {label}"

---

## 8.4 `comparison_plots.py` — PCA vs Autoencoder Side-by-Side

### `plot_cumulative_pnl_comparison(pca_returns, ae_returns, pca_returns_nc, ae_returns_nc, save_path=None)`
- Single chart with 4 lines:
  - PCA with costs (solid blue)
  - AE with costs (solid red)
  - PCA without costs (dashed blue)
  - AE without costs (dashed red)
- Title: "Cumulative PnL: PCA vs Autoencoder"

### `plot_metrics_comparison_bar(pca_metrics, ae_metrics, save_path=None)`
- Grouped bar chart of key metrics side by side:
  - Sharpe ratio, max drawdown (absolute), hit ratio, annualized turnover
  - Show both with-cost and without-cost versions
- Title: "Performance Metrics: PCA vs Autoencoder"

### `plot_grid_search_heatmap(results_df, model_type, metric='sharpe_with_costs', save_path=None)`
- For PCA: 2D heatmap of Sharpe ratio vs (n_factors × zscore_entry)
- For AE: 2D heatmap of Sharpe ratio vs (bottleneck × activation), one panel per depth
- Color scale: diverging, centered at 0 Sharpe
- Title: f"Grid Search Results — {model_type.upper()}: {metric}"

### `plot_rolling_sharpe_comparison(pca_returns, ae_returns, window=63, save_path=None)`
- Overlay rolling Sharpe (PCA blue, AE red) on same chart
- Shade region where AE outperforms PCA (light red) and where PCA wins (light blue)
- Title: f"Rolling {window}-Day Sharpe Ratio Comparison"

### `plot_correlation_of_returns(pca_returns, ae_returns, save_path=None)`
- Scatter plot of daily returns (PCA x-axis, AE y-axis)
- Add regression line and Pearson correlation annotation
- Title: "Correlation of Daily Returns: PCA vs Autoencoder"

### `create_full_comparison_dashboard(pca_results, ae_results, pca_metrics, ae_metrics, save_path=None)`
- A large multi-panel figure (3×3 or 4×2 grid) combining:
  - Cumulative PnL comparison
  - Drawdown comparison
  - Rolling Sharpe comparison
  - Metrics bar chart
  - Monthly returns heatmap (PCA)
  - Monthly returns heatmap (AE)
  - Position count comparison
  - Transaction cost comparison
- This is the "executive summary" visualization — make it publication quality
- Title: "Statistical Arbitrage: PCA vs Deep Autoencoder — Full Comparison Dashboard"
- Figure size: 20×16 inches minimum
