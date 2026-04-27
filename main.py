"""
main.py — PCA_project pipeline entry point.

Run the full statistical arbitrage pipeline end-to-end:
  data download → PCA grid search → AE grid search →
  best-model fitting → backtesting → metrics export

Usage:
    python main.py                        # run all stages, skip cached
    python main.py --force                # re-run all stages, overwrite cache
    python main.py --stage data           # run only the data stage
    python main.py --stage pca            # run only the PCA grid search stage
    python main.py --stage ae             # run only the AE grid search stage
    python main.py --stage backtest       # run only the backtest stage
    python main.py --stage metrics        # run only the metrics stage

All results are exported to data/results/ as .pkl files.
Open notebooks/analysis.ipynb after running this script to view visualizations.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

from pca_project import load_config
from pca_project.backtesting.engine import run_full_backtest
from pca_project.data.downloader import PriceDownloader
from pca_project.data.preprocessor import DataPreprocessor
from pca_project.data.universe import SP500Universe
from pca_project.experiments import (
    load_results,
    results_exist,
    save_results,
)
from pca_project.experiments.ae_grid_search import AEGridSearch
from pca_project.experiments.pca_grid_search import PCAGridSearch
from pca_project.factors.autoencoder_model import AutoencoderModel
from pca_project.factors.pca_model import PCAModel
from pca_project.metrics.performance import PerformanceAnalyzer


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_data(config: dict, force: bool) -> dict:
    """Download prices, compute log returns, split, and standardize.

    Returns:
        Data dict from DataPreprocessor.run().
    """
    if not force and results_exist("data_splits.pkl", config):
        logging.info("Stage [data]: cached results found, loading from disk.")
        return load_results("data_splits.pkl", config)

    logging.info("Stage [data]: running...")
    t0 = time.perf_counter()

    universe = SP500Universe(config)
    tickers = universe.get_tickers()

    downloader = PriceDownloader(config)
    prices = downloader.download(
        tickers,
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
    )

    preprocessor = DataPreprocessor(config)
    data = preprocessor.run(prices)

    save_results(data, "data_splits.pkl", config)
    logging.info(
        "Stage [data]: complete in %.1fs. %d assets × %d days total.",
        time.perf_counter() - t0,
        data["raw_returns"].shape[1],
        data["raw_returns"].shape[0],
    )
    return data


def stage_pca_grid_search(config: dict, data: dict, force: bool) -> pd.DataFrame:
    """Run PCA grid search over n_factors × zscore thresholds.

    Returns:
        DataFrame of all grid-search results.
    """
    if not force and results_exist("pca_grid_search.pkl", config):
        logging.info("Stage [pca_grid_search]: cached results found, loading.")
        return load_results("pca_grid_search.pkl", config)

    logging.info("Stage [pca_grid_search]: running...")
    t0 = time.perf_counter()

    gs = PCAGridSearch(config)
    results_df = gs.run(data, verbose=True)

    save_results(results_df, "pca_grid_search.pkl", config)
    logging.info(
        "Stage [pca_grid_search]: complete in %.1fs. %d configurations.",
        time.perf_counter() - t0,
        len(results_df),
    )
    return results_df


def stage_ae_grid_search(config: dict, data: dict, force: bool) -> pd.DataFrame:
    """Run AE grid search over bottleneck × depth × activation (default bottleneck).

    Returns:
        DataFrame of all grid-search results.
    """
    if not force and results_exist("ae_grid_search.pkl", config):
        logging.info("Stage [ae_grid_search]: cached results found, loading.")
        return load_results("ae_grid_search.pkl", config)

    logging.info("Stage [ae_grid_search]: running (default bottleneck, vary depth & activation)...")
    t0 = time.perf_counter()

    gs = AEGridSearch(config)
    results_df = gs.run(data, full_grid=False, verbose=True)

    save_results(results_df, "ae_grid_search.pkl", config)
    logging.info(
        "Stage [ae_grid_search]: complete in %.1fs. %d configurations.",
        time.perf_counter() - t0,
        len(results_df),
    )
    return results_df


def stage_fit_best_models(
    config: dict,
    data: dict,
    pca_results: pd.DataFrame,
    ae_results: pd.DataFrame,
    force: bool,
) -> tuple[PCAModel, AutoencoderModel]:
    """Fit the best PCA and AE models identified from grid search.

    Returns:
        Tuple of (best_pca_model, best_ae_model).
    """
    pca_cached = not force and results_exist("pca_best_model.pkl", config)
    ae_cached = not force and results_exist("ae_best_model.pkl", config)

    if pca_cached:
        logging.info("Stage [fit_best_pca]: loading from cache.")
        pca_model = load_results("pca_best_model.pkl", config)
    else:
        logging.info("Stage [fit_best_pca]: fitting best PCA model...")
        t0 = time.perf_counter()
        gs_pca = PCAGridSearch(config)
        best_pca_cfg = gs_pca.get_best_config(pca_results)
        logging.info("Best PCA config: %s", best_pca_cfg)

        pca_model = PCAModel(config, n_factors=best_pca_cfg["n_factors"])
        pca_model.fit(data["train_std"])
        save_results(pca_model, "pca_best_model.pkl", config)
        logging.info("Stage [fit_best_pca]: done in %.1fs.", time.perf_counter() - t0)

    if ae_cached:
        logging.info("Stage [fit_best_ae]: loading from cache.")
        ae_model = load_results("ae_best_model.pkl", config)
    else:
        logging.info("Stage [fit_best_ae]: fitting best AE model...")
        t0 = time.perf_counter()
        gs_ae = AEGridSearch(config)
        best_ae_cfg = gs_ae.get_best_config(ae_results)
        logging.info("Best AE config: %s", best_ae_cfg)

        ae_model = AutoencoderModel(
            config,
            bottleneck=best_ae_cfg["bottleneck"],
            depth=best_ae_cfg["depth"],
            activation=best_ae_cfg["activation"],
        )
        ae_model.fit(data["train_std"], data["val_std"])
        save_results(ae_model, "ae_best_model.pkl", config)
        logging.info("Stage [fit_best_ae]: done in %.1fs.", time.perf_counter() - t0)

    return pca_model, ae_model


def stage_backtest(
    config: dict,
    data: dict,
    pca_model: PCAModel,
    ae_model: AutoencoderModel,
    pca_results: pd.DataFrame,
    ae_results: pd.DataFrame,
    force: bool,
) -> tuple[dict, dict]:
    """Run full backtests for both models using their best Z-score thresholds.

    Returns:
        Tuple of (pca_backtest_results, ae_backtest_results).
    """
    pca_cached = not force and results_exist("pca_backtest.pkl", config)
    ae_cached = not force and results_exist("ae_backtest.pkl", config)

    gs_pca = PCAGridSearch(config)
    best_pca_cfg = gs_pca.get_best_config(pca_results)
    gs_ae = AEGridSearch(config)
    best_ae_cfg = gs_ae.get_best_config(ae_results)

    if pca_cached:
        logging.info("Stage [backtest_pca]: loading from cache.")
        pca_bt = load_results("pca_backtest.pkl", config)
    else:
        logging.info("Stage [backtest_pca]: running...")
        t0 = time.perf_counter()
        pca_bt = run_full_backtest(
            pca_model,
            data["test_raw"],
            data["test_std"],
            config,
            zscore_entry=best_pca_cfg["zscore_entry"],
            zscore_exit=best_pca_cfg["zscore_exit"],
        )
        save_results(pca_bt["signals"], "pca_signals.pkl", config)
        save_results(pca_bt["zscores"], "pca_zscores.pkl", config)
        save_results(pca_bt, "pca_backtest.pkl", config)
        logging.info("Stage [backtest_pca]: done in %.1fs.", time.perf_counter() - t0)

    if ae_cached:
        logging.info("Stage [backtest_ae]: loading from cache.")
        ae_bt = load_results("ae_backtest.pkl", config)
    else:
        logging.info("Stage [backtest_ae]: running...")
        t0 = time.perf_counter()
        ae_bt = run_full_backtest(
            ae_model,
            data["test_raw"],
            data["test_std"],
            config,
            zscore_entry=best_ae_cfg["zscore_entry"],
            zscore_exit=best_ae_cfg["zscore_exit"],
        )
        save_results(ae_bt["signals"], "ae_signals.pkl", config)
        save_results(ae_bt["zscores"], "ae_zscores.pkl", config)
        save_results(ae_bt, "ae_backtest.pkl", config)
        logging.info("Stage [backtest_ae]: done in %.1fs.", time.perf_counter() - t0)

    return pca_bt, ae_bt


def stage_metrics(
    config: dict,
    pca_bt: dict,
    ae_bt: dict,
    force: bool,
) -> tuple[dict, dict]:
    """Compute and save performance metrics for both models.

    Returns:
        Tuple of (pca_metrics_dict, ae_metrics_dict), each containing
        ``with_costs`` and ``without_costs`` metric dicts.
    """
    if not force and results_exist("pca_metrics.pkl", config) and results_exist("ae_metrics.pkl", config):
        logging.info("Stage [metrics]: loading from cache.")
        return load_results("pca_metrics.pkl", config), load_results("ae_metrics.pkl", config)

    logging.info("Stage [metrics]: computing...")
    t0 = time.perf_counter()
    analyzer = PerformanceAnalyzer(config)

    pca_metrics = {
        "with_costs": analyzer.compute_all(
            pca_bt["with_costs"]["daily_returns"], pca_bt["with_costs"]["weights"]
        ),
        "without_costs": analyzer.compute_all(
            pca_bt["without_costs"]["daily_returns"], pca_bt["without_costs"]["weights"]
        ),
    }
    ae_metrics = {
        "with_costs": analyzer.compute_all(
            ae_bt["with_costs"]["daily_returns"], ae_bt["with_costs"]["weights"]
        ),
        "without_costs": analyzer.compute_all(
            ae_bt["without_costs"]["daily_returns"], ae_bt["without_costs"]["weights"]
        ),
    }

    save_results(pca_metrics, "pca_metrics.pkl", config)
    save_results(ae_metrics, "ae_metrics.pkl", config)

    logging.info("Stage [metrics]: done in %.1fs.", time.perf_counter() - t0)
    logging.info(
        "PCA  Sharpe (w/ costs): %.4f | AE Sharpe (w/ costs): %.4f",
        pca_metrics["with_costs"]["sharpe_ratio"],
        ae_metrics["with_costs"]["sharpe_ratio"],
    )
    return pca_metrics, ae_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full pipeline end-to-end."""
    parser = argparse.ArgumentParser(
        description="PCA_project — Statistical Arbitrage Pipeline"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages, overwriting cached results.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "data", "pca", "ae", "backtest", "metrics"],
        help="Run only a specific stage.",
    )
    args = parser.parse_args()

    config = load_config()

    logging.basicConfig(
        level=getattr(logging, config["log_level"]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Global reproducibility seeds
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    logging.info("=" * 60)
    logging.info("PCA_project — Statistical Arbitrage Pipeline")
    logging.info("=" * 60)

    pipeline_start = time.perf_counter()

    if args.stage in ("all", "data"):
        data = stage_data(config, args.force)
    else:
        data = load_results("data_splits.pkl", config)

    if args.stage in ("all", "pca"):
        pca_results = stage_pca_grid_search(config, data, args.force)
    else:
        pca_results = load_results("pca_grid_search.pkl", config)

    if args.stage in ("all", "ae"):
        ae_results = stage_ae_grid_search(config, data, args.force)
    else:
        ae_results = load_results("ae_grid_search.pkl", config)

    if args.stage in ("all", "backtest", "metrics"):
        pca_model, ae_model = stage_fit_best_models(
            config, data, pca_results, ae_results, args.force
        )
        pca_bt, ae_bt = stage_backtest(
            config, data, pca_model, ae_model, pca_results, ae_results, args.force
        )
    elif args.stage == "metrics":
        pca_bt = load_results("pca_backtest.pkl", config)
        ae_bt = load_results("ae_backtest.pkl", config)

    if args.stage in ("all", "metrics"):
        stage_metrics(config, pca_bt, ae_bt, args.force)

    logging.info("=" * 60)
    logging.info(
        "Pipeline complete in %.1fs. All results saved to data/results/",
        time.perf_counter() - pipeline_start,
    )
    logging.info("Open notebooks/analysis.ipynb to view visualizations.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
