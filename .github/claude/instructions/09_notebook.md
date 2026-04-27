# Instruction 09 — analysis.ipynb (Single Visualization Notebook)

The project has exactly **one notebook**: `notebooks/analysis.ipynb`.

Its sole purpose is to load the results exported by `main.py` and produce all visualizations. It must never run experiments, fit models, or compute residuals. It is a read-and-plot layer only.

---

## Hard Rules for This Notebook

- **Never import** `PCAGridSearch`, `AEGridSearch`, `BacktestEngine`, `DataPreprocessor`, `PCAModel`, or `AutoencoderModel` — these belong to `main.py` only.
- **Always** check for results on disk before any section. If a required file is missing, raise a clear `FileNotFoundError` with the message: `"Results not found. Run `python main.py` first."`
- Every plot must be saved to `data/results/figures/` in addition to being displayed inline.
- The notebook must be fully re-runnable from top to bottom in a fresh kernel without errors, provided `main.py` has been run first.

---

## Notebook Structure

### Cell 1 — Markdown: Header

```markdown
# PCA_project — Results Analysis

This notebook visualizes the results of the statistical arbitrage pipeline.
**Prerequisite**: run `python main.py` from the project root before opening this notebook.
```

### Cell 2 — Imports and Setup

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pca_project import load_config
from pca_project.experiments import load_results, results_exist
from pca_project.metrics.performance import PerformanceAnalyzer
from pca_project.visualization.factor_plots import (
    plot_eigenvalue_spectrum,
    plot_explained_variance_vs_k,
    plot_eigenvector_weights,
    plot_autoencoder_loss_curves,
    plot_reconstruction_quality,
)
from pca_project.visualization.signal_plots import (
    plot_zscore_timeseries,
    plot_residual_acf,
    plot_ou_parameter_distribution,
    plot_signal_heatmap,
    plot_position_counts,
)
from pca_project.visualization.backtest_plots import (
    plot_cumulative_pnl,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_monthly_returns_heatmap,
    plot_transaction_costs_impact,
)
from pca_project.visualization.comparison_plots import (
    plot_cumulative_pnl_comparison,
    plot_metrics_comparison_bar,
    plot_grid_search_heatmap,
    plot_rolling_sharpe_comparison,
    plot_correlation_of_returns,
    create_full_comparison_dashboard,
)

config = load_config()
np.random.seed(config["random_seed"])

RESULTS_DIR = Path(config["data"]["results_dir"])
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results directory: {RESULTS_DIR}")
print(f"Figures will be saved to: {FIGURES_DIR}")
```

### Cell 3 — Load All Results

```python
# Fail fast with a clear message if main.py has not been run
required_files = [
    "data_splits.pkl", "pca_grid_search.pkl", "ae_grid_search.pkl",
    "pca_backtest.pkl", "ae_backtest.pkl",
    "pca_metrics.pkl", "ae_metrics.pkl",
    "pca_signals.pkl", "ae_signals.pkl",
    "pca_zscores.pkl", "ae_zscores.pkl",
]
for f in required_files:
    if not results_exist(f, config):
        raise FileNotFoundError(
            f"Required results file '{f}' not found. "
            "Run `python main.py` from the project root first."
        )

data         = load_results("data_splits.pkl", config)
pca_grid     = load_results("pca_grid_search.pkl", config)
ae_grid      = load_results("ae_grid_search.pkl", config)
pca_bt       = load_results("pca_backtest.pkl", config)
ae_bt        = load_results("ae_backtest.pkl", config)
pca_metrics  = load_results("pca_metrics.pkl", config)
ae_metrics   = load_results("ae_metrics.pkl", config)
pca_signals  = load_results("pca_signals.pkl", config)
ae_signals   = load_results("ae_signals.pkl", config)
pca_zscores  = load_results("pca_zscores.pkl", config)
ae_zscores   = load_results("ae_zscores.pkl", config)

analyzer = PerformanceAnalyzer(config)
print("All results loaded successfully.")
```

---

## Sections & Plots

### Section 1 — Markdown: Data Overview

```markdown
## 1. Data Overview
```

Plots and tables to include:
- Print the data split summary table: split name, start date, end date, n_days, n_stocks
- Plot cross-sectional mean and std of raw returns over time
- Distribution of daily log returns before vs after cross-sectional standardization (side-by-side histograms)
- Correlation matrix heatmap of a sample of 50 stocks (training set)

---

### Section 2 — Markdown: PCA Factor Analysis

```markdown
## 2. PCA Factor Analysis
```

Plots and tables:
- `plot_eigenvalue_spectrum()` — with markdown explanation before the plot
- `plot_explained_variance_vs_k()` — show 45/55/65% threshold lines
- Print variance explained table: k → cumulative variance %
- `plot_eigenvector_weights()` for the first 3 eigenvectors — with brief interpretation in markdown

---

### Section 3 — Markdown: Autoencoder Analysis

```markdown
## 3. Autoencoder Analysis
```

Plots and tables:
- `plot_autoencoder_loss_curves()` — training and validation loss for the best model
- `plot_reconstruction_quality()` for 3 representative stocks
- Overlay loss curves for each activation function (tanh, elu, relu) on a single chart
- Table: validation loss and test Sharpe for each activation function

---

### Section 4 — Markdown: Grid Search Results

```markdown
## 4. Hyperparameter Grid Search
```

Plots and tables:
- `plot_grid_search_heatmap()` for PCA — Sharpe vs (n_factors × zscore_entry)
- `plot_grid_search_heatmap()` for AE — Sharpe vs (bottleneck × activation), one panel per depth
- Top 10 PCA configurations by Sharpe (with costs) as a formatted DataFrame
- Top 10 AE configurations by Sharpe (with costs) as a formatted DataFrame
- Best configuration for each model, clearly highlighted

---

### Section 5 — Markdown: Signal Analysis

```markdown
## 5. Signal Analysis
```

Plots and tables:
- `plot_signal_heatmap()` for PCA — overview of positions across the test period
- `plot_signal_heatmap()` for AE
- `plot_position_counts()` for both models (separate charts)
- `plot_zscore_timeseries()` for 3 hand-picked stocks — show entry, exit, and OU behavior
- `plot_ou_parameter_distribution()` — kappa distribution, showing the fraction of stocks filtered

---

### Section 6 — Markdown: PCA Backtest Results

```markdown
## 6. PCA Strategy — Backtest Results
```

Plots and tables:
- `plot_cumulative_pnl()` — with and without costs, gap shaded
- `plot_drawdown()` — with costs
- `plot_rolling_sharpe()` — 63-day window
- `plot_monthly_returns_heatmap()` — full test period calendar view
- `plot_transaction_costs_impact()` — three-panel cost analysis
- Print all metrics from `pca_metrics` as a formatted table (both with and without costs)

---

### Section 7 — Markdown: Autoencoder Backtest Results

```markdown
## 7. Autoencoder Strategy — Backtest Results
```

Same plots as Section 6, but for the AE strategy:
- `plot_cumulative_pnl()`
- `plot_drawdown()`
- `plot_rolling_sharpe()`
- `plot_monthly_returns_heatmap()`
- `plot_transaction_costs_impact()`
- Print `ae_metrics` as a formatted table

---

### Section 8 — Markdown: Head-to-Head Comparison

```markdown
## 8. PCA vs Autoencoder — Comparison
```

This is the most important section. Include all of:
- `plot_cumulative_pnl_comparison()` — 4 lines (each model × with/without costs)
- `plot_rolling_sharpe_comparison()` — shaded regions showing which model leads
- `plot_metrics_comparison_bar()` — grouped bars for Sharpe, MDD, hit ratio, turnover
- `plot_correlation_of_returns()` — scatter of daily returns with Pearson r annotation
- `PerformanceAnalyzer.compare()` — formatted side-by-side metrics table; bold the winner in each row
- Bar chart: "Alpha destroyed by transaction costs" = Sharpe_no_cost − Sharpe_with_cost for each model

---

### Section 9 — Markdown: Full Dashboard

```markdown
## 9. Summary Dashboard
```

- `create_full_comparison_dashboard()` — the publication-quality multi-panel summary figure
- Save the figure to `data/results/figures/comparison_dashboard.png` at high DPI (300)
- Final markdown cell with written conclusions:
  - Which model performs better overall and why
  - In which market regimes does each model have an edge
  - Impact of transaction costs on each model
  - Limitations and directions for future work
