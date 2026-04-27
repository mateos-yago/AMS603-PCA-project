# Instruction 07 — Experiments, Result Export & main.py

All experiment orchestration lives in `pca_project/experiments/`. The single entry point for running everything is `main.py` at the project root.

---

## 7.1 Experiment Result Schema

Every configuration tested must produce a result dict saved to disk. Define a dataclass for this in `pca_project/experiments/__init__.py`:

```python
@dataclass
class ExperimentResult:
    # Identification
    model_type: str            # "pca" or "autoencoder"
    experiment_id: str         # unique hash or timestamp-based ID
    timestamp: str

    # Hyperparameters
    n_factors: int             # bottleneck size (for AE) or k (for PCA)
    zscore_entry: float
    zscore_exit: float
    # AE-specific (None for PCA)
    depth: int | None
    activation: str | None

    # Variance explained (PCA) or reconstruction loss (AE)
    variance_explained: float | None
    final_val_loss: float | None

    # Performance with transaction costs
    sharpe_with_costs: float
    max_drawdown_with_costs: float
    hit_ratio_with_costs: float
    annualized_return_with_costs: float
    annualized_turnover_with_costs: float

    # Performance without transaction costs
    sharpe_without_costs: float
    max_drawdown_without_costs: float
    hit_ratio_without_costs: float
    annualized_return_without_costs: float
    annualized_turnover_without_costs: float
```

---

## 7.2 `pca_grid_search.py` — `PCAGridSearch`

### Class: `PCAGridSearch`

```
PCAGridSearch(config: dict)
```

### Method: `run(self, data: dict, verbose: bool = True) -> pd.DataFrame`

**Input**: `data` dict from `DataPreprocessor.run()`

**Grid**: Cross-product of `config['pca']['n_factors_grid']` × `config['signals']['zscore_entry_grid']` × `config['signals']['zscore_exit_grid']`

**For each configuration**:
1. Instantiate `PCAModel(config, n_factors=k)`
2. `model.fit(data['train_std'])`
3. Log `variance_explained` for this k value
4. Get residuals on test set: `model.get_residuals(data['test_std'])`
5. Run `run_full_backtest(model, data['test_raw'], data['test_std'], config, zscore_entry, zscore_exit)`
6. Compute all metrics using `PerformanceAnalyzer`
7. Store as `ExperimentResult`

**Parallelization**: Use `joblib.Parallel(n_jobs=config['experiments']['n_jobs'])` with `delayed`.

**Output**: DataFrame of all results.

### Method: `get_best_config(self, results_df: pd.DataFrame, metric: str = "sharpe_with_costs") -> dict`

Returns the hyperparameter dict corresponding to the row with the highest value of `metric`.

---

## 7.3 `ae_grid_search.py` — `AEGridSearch`

### Class: `AEGridSearch`

```
AEGridSearch(config: dict)
```

### Method: `run(self, data: dict, verbose: bool = True) -> pd.DataFrame`

**Grid**: Cross-product of:
- `config['autoencoder']['bottleneck_grid']`
- `config['autoencoder']['depth_grid']`
- `config['autoencoder']['activation_grid']`
- `config['signals']['zscore_entry_grid']`
- `config['signals']['zscore_exit_grid']`

**For each configuration**:
1. Instantiate `AutoencoderModel(config, bottleneck=k, depth=d, activation=act)`
2. `model.fit(data['train_std'], data['val_std'])` — val set used for early stopping
3. Get residuals on test set
4. Run full backtest
5. Store `ExperimentResult` including `final_val_loss`

**Important**: Allow a `full_grid: bool = False` parameter. When False, only run the default hyperparameter values from config (useful for a quick first pass). Log a warning when `full_grid=True` about expected compute time.

---

## 7.4 Result Persistence

### Functions in `pca_project/experiments/__init__.py`:

```python
def save_results(results: dict, filename: str, config: dict) -> None:
    """
    Serialize and save a results dict to data/results/<filename>.pkl.
    Also save any DataFrames inside the dict as companion CSV files
    for easy inspection without Python.
    """

def load_results(filename: str, config: dict) -> dict:
    """Load a previously saved results dict from data/results/<filename>.pkl."""

def results_exist(filename: str, config: dict) -> bool:
    """Return True if data/results/<filename>.pkl exists on disk."""
```

### What gets saved

`main.py` saves the following files to `data/results/`:

| Filename | Contents |
|----------|----------|
| `data_splits.pkl` | The full `data` dict from `DataPreprocessor.run()` |
| `pca_grid_search.pkl` | DataFrame of all PCA grid search results |
| `ae_grid_search.pkl` | DataFrame of all AE grid search results |
| `pca_best_model.pkl` | Fitted `PCAModel` instance (best config) |
| `ae_best_model.pkl` | Fitted `AutoencoderModel` instance (best config) |
| `pca_backtest.pkl` | Full backtest result dict for best PCA config |
| `ae_backtest.pkl` | Full backtest result dict for best AE config |
| `pca_metrics.pkl` | `PerformanceAnalyzer.compute_all()` output for PCA |
| `ae_metrics.pkl` | `PerformanceAnalyzer.compute_all()` output for AE |
| `pca_signals.pkl` | Signal DataFrame for the test period (PCA) |
| `ae_signals.pkl` | Signal DataFrame for the test period (AE) |
| `pca_zscores.pkl` | Z-score DataFrame for the test period (PCA) |
| `ae_zscores.pkl` | Z-score DataFrame for the test period (AE) |

Every `.pkl` file that contains a DataFrame should also be saved as a companion `.csv` so results can be inspected without loading Python.

---

## 7.5 `main.py` — The Single Entry Point

`main.py` lives at the **project root** (not inside the package). It orchestrates the entire pipeline from data download to result export. The notebook never calls anything in this file — it only reads the outputs.

### Structure of `main.py`

```python
"""
main.py — PCA_project pipeline entry point.

Usage:
    python main.py              # run all stages, skip cached results
    python main.py --force      # re-run all stages, overwrite cached results
    python main.py --stage data # run only the data stage
"""

import argparse
import logging
import numpy as np
import torch

from pca_project import load_config
from pca_project.data.universe import SP500Universe
from pca_project.data.downloader import PriceDownloader
from pca_project.data.preprocessor import DataPreprocessor
from pca_project.experiments.pca_grid_search import PCAGridSearch
from pca_project.experiments.ae_grid_search import AEGridSearch
from pca_project.experiments import save_results, load_results, results_exist
from pca_project.factors.pca_model import PCAModel
from pca_project.factors.autoencoder_model import AutoencoderModel
from pca_project.backtesting.engine import run_full_backtest
from pca_project.metrics.performance import PerformanceAnalyzer
```

### Stages

Implement each stage as a standalone function. `main()` calls them in order, checking the cache before each one:

```python
def stage_data(config: dict, force: bool) -> dict:
    """Download prices, compute returns, split, standardize. Returns data dict."""

def stage_pca_grid_search(config: dict, data: dict, force: bool) -> pd.DataFrame:
    """Run PCA grid search. Returns results DataFrame."""

def stage_ae_grid_search(config: dict, data: dict, force: bool) -> pd.DataFrame:
    """Run AE grid search. Returns results DataFrame."""

def stage_fit_best_models(config: dict, data: dict,
                          pca_results: pd.DataFrame,
                          ae_results: pd.DataFrame,
                          force: bool) -> tuple:
    """Fit the best PCA and AE models. Returns (pca_model, ae_model)."""

def stage_backtest(config: dict, data: dict,
                   pca_model: PCAModel,
                   ae_model: AutoencoderModel,
                   force: bool) -> tuple:
    """Run backtests for both models. Returns (pca_bt_results, ae_bt_results)."""

def stage_metrics(config: dict,
                  pca_bt: dict, ae_bt: dict,
                  force: bool) -> tuple:
    """Compute and save performance metrics. Returns (pca_metrics, ae_metrics)."""
```

### Cache logic (apply consistently to every stage)

```python
def stage_data(config, force):
    if not force and results_exist("data_splits.pkl", config):
        logging.info("Stage [data]: cached results found, loading from disk.")
        return load_results("data_splits.pkl", config)

    logging.info("Stage [data]: running...")
    # ... actual computation ...
    save_results(data, "data_splits.pkl", config)
    return data
```

### `main()` function

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages, overwriting cached results.")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "data", "pca", "ae", "backtest", "metrics"],
                        help="Run only a specific stage.")
    args = parser.parse_args()

    config = load_config()

    # Set global random seeds for reproducibility
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    logging.basicConfig(level=config["log_level"],
                        format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info("=" * 60)
    logging.info("PCA_project — Statistical Arbitrage Pipeline")
    logging.info("=" * 60)

    data         = stage_data(config, args.force)
    pca_results  = stage_pca_grid_search(config, data, args.force)
    ae_results   = stage_ae_grid_search(config, data, args.force)
    pca_model, ae_model = stage_fit_best_models(config, data, pca_results, ae_results, args.force)
    pca_bt, ae_bt       = stage_backtest(config, data, pca_model, ae_model, args.force)
    pca_metrics, ae_metrics = stage_metrics(config, pca_bt, ae_bt, args.force)

    logging.info("=" * 60)
    logging.info("Pipeline complete. All results saved to data/results/")
    logging.info("Open notebooks/analysis.ipynb to view visualizations.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
```

### Logging

Every stage must log:
- When it starts
- Whether it loaded from cache or computed fresh
- Key summary statistics on completion (e.g., "PCA grid search: 60 configs tested, best Sharpe = 1.34")
- How long it took (use `time.perf_counter()`)
