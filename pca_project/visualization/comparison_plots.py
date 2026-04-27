"""Visualization functions for PCA vs Autoencoder comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

plt.style.use("seaborn-v0_8-whitegrid")

PCA_COLOR = "#2E86AB"
AE_COLOR = "#E84855"


def _save(fig: Figure, save_path: str | None) -> Figure:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _cum(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def _dd(r: pd.Series) -> pd.Series:
    wealth = _cum(r)
    return (wealth - wealth.cummax()) / wealth.cummax()


def plot_cumulative_pnl_comparison(
    pca_returns: pd.Series,
    ae_returns: pd.Series,
    pca_returns_nc: pd.Series,
    ae_returns_nc: pd.Series,
    save_path: str | None = None,
) -> Figure:
    """Four-line cumulative PnL comparison chart.

    Args:
        pca_returns: PCA daily returns with costs.
        ae_returns: AE daily returns with costs.
        pca_returns_nc: PCA daily returns without costs.
        ae_returns_nc: AE daily returns without costs.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(_cum(pca_returns).values, color=PCA_COLOR, linewidth=2.2, label="PCA (with costs)")
    ax.plot(_cum(ae_returns).values, color=AE_COLOR, linewidth=2.2, label="AE (with costs)")
    ax.plot(_cum(pca_returns_nc).values, color=PCA_COLOR, linewidth=1.2, linestyle="--", label="PCA (no costs)")
    ax.plot(_cum(ae_returns_nc).values, color=AE_COLOR, linewidth=1.2, linestyle="--", label="AE (no costs)")
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")

    # Use PCA index for x-ticks
    idx = pca_returns.index
    n_ticks = min(8, len(idx))
    tick_locs = np.linspace(0, len(idx) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([str(idx[i].date()) for i in tick_locs], rotation=30, fontsize=8)
    ax.set_ylabel("Cumulative Return (indexed to 1)")
    ax.set_title("Cumulative PnL: PCA vs Autoencoder")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_metrics_comparison_bar(
    pca_metrics: dict,
    ae_metrics: dict,
    save_path: str | None = None,
) -> Figure:
    """Grouped bar chart of key performance metrics.

    Args:
        pca_metrics: Dict with ``with_costs`` and ``without_costs`` sub-dicts.
        ae_metrics: Same structure for autoencoder.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    metrics_to_show = [
        ("sharpe_ratio", "Sharpe Ratio"),
        ("maximum_drawdown", "Max Drawdown"),
        ("hit_ratio", "Hit Ratio"),
        ("annualized_return", "Ann. Return"),
    ]
    scenarios = [("with_costs", "With Costs"), ("without_costs", "No Costs")]

    n_metrics = len(metrics_to_show)
    n_groups = n_metrics
    x = np.arange(n_groups)
    bar_width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(13, 5))
    color_map = {
        ("pca", "with_costs"): PCA_COLOR,
        ("pca", "without_costs"): "#7BBCCF",
        ("ae", "with_costs"): AE_COLOR,
        ("ae", "without_costs"): "#F09099",
    }

    for i, (scen_key, scen_label) in enumerate(scenarios):
        for j, (model_key, model_metrics, model_label) in enumerate(
            [("pca", pca_metrics, "PCA"), ("ae", ae_metrics, "AE")]
        ):
            vals = [model_metrics[scen_key].get(m, 0) for m, _ in metrics_to_show]
            offset = offsets[i * 2 + j] * bar_width
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_width,
                color=color_map[(model_key, scen_key)],
                label=f"{model_label} ({scen_label})",
                alpha=0.85,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics_to_show])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Performance Metrics: PCA vs Autoencoder")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_grid_search_heatmap(
    results_df: pd.DataFrame,
    model_type: str,
    metric: str = "sharpe_with_costs",
    save_path: str | None = None,
) -> Figure:
    """2-D heatmap of grid search Sharpe ratios.

    For PCA: n_factors × zscore_entry.
    For AE: bottleneck × activation (one panel per depth).

    Args:
        results_df: Grid search results DataFrame.
        model_type: ``"pca"`` or ``"autoencoder"``.
        metric: Column to display as heatmap values.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    if model_type == "pca":
        pivot = results_df.pivot_table(
            values=metric, index="n_factors", columns="zscore_entry", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01) if pivot.size else 1
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Z-score Entry Threshold")
        ax.set_ylabel("n_factors")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label=metric)
        ax.set_title(f"Grid Search Results — PCA: {metric}")
        fig.tight_layout()
    else:
        depths = sorted(results_df["depth"].dropna().unique())
        fig, axes = plt.subplots(1, len(depths), figsize=(6 * len(depths), 5))
        if len(depths) == 1:
            axes = [axes]
        for ax, d in zip(axes, depths):
            sub = results_df[results_df["depth"] == d]
            pivot = sub.pivot_table(
                values=metric, index="n_factors", columns="activation", aggfunc="mean"
            )
            vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01) if pivot.size else 1
            im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            ax.set_xlabel("Activation")
            ax.set_ylabel("Bottleneck")
            ax.set_title(f"Depth={int(d)}")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax, label=metric)
        fig.suptitle(f"Grid Search Results — AE: {metric}", fontsize=12)
        fig.tight_layout()
    return _save(fig, save_path)


def plot_rolling_sharpe_comparison(
    pca_returns: pd.Series,
    ae_returns: pd.Series,
    window: int = 63,
    save_path: str | None = None,
) -> Figure:
    """Overlay rolling Sharpe with region shading.

    Args:
        pca_returns: PCA daily returns.
        ae_returns: AE daily returns.
        window: Rolling window in days.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    def _rolling_sharpe(r: pd.Series) -> pd.Series:
        rm = r.rolling(window).mean()
        rs = r.rolling(window).std(ddof=1)
        return (rm / rs * np.sqrt(252)).fillna(0)

    pca_rs = _rolling_sharpe(pca_returns)
    ae_rs = _rolling_sharpe(ae_returns)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pca_rs.index, pca_rs.values, color=PCA_COLOR, linewidth=1.8, label="PCA")
    ax.plot(ae_rs.index, ae_rs.values, color=AE_COLOR, linewidth=1.8, label="AE")
    ae_wins = ae_rs.values > pca_rs.values
    pca_wins = ~ae_wins
    ax.fill_between(pca_rs.index, pca_rs.values, ae_rs.values,
                    where=ae_wins, alpha=0.18, color=AE_COLOR, label="AE outperforms")
    ax.fill_between(pca_rs.index, pca_rs.values, ae_rs.values,
                    where=pca_wins, alpha=0.18, color=PCA_COLOR, label="PCA outperforms")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color="green", linewidth=0.8, linestyle=":", label="Sharpe=1")
    ax.set_ylabel("Rolling Sharpe Ratio (annualised)")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_correlation_of_returns(
    pca_returns: pd.Series,
    ae_returns: pd.Series,
    save_path: str | None = None,
) -> Figure:
    """Scatter plot of daily returns with regression line and Pearson r.

    Args:
        pca_returns: PCA daily returns.
        ae_returns: AE daily returns.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    common = pca_returns.index.intersection(ae_returns.index)
    x = pca_returns.loc[common].values
    y = ae_returns.loc[common].values
    corr = float(np.corrcoef(x, y)[0, 1])

    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.3, s=8, color=AE_COLOR)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5, label=f"Regression line")
    ax.annotate(f"Pearson r = {corr:.3f}", xy=(0.05, 0.90), xycoords="axes fraction", fontsize=11)
    ax.set_xlabel("PCA Daily Returns")
    ax.set_ylabel("AE Daily Returns")
    ax.set_title("Correlation of Daily Returns: PCA vs Autoencoder")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def create_full_comparison_dashboard(
    pca_results: dict,
    ae_results: dict,
    pca_metrics: dict,
    ae_metrics: dict,
    save_path: str | None = None,
) -> Figure:
    """Publication-quality 4×2 comparison dashboard.

    Args:
        pca_results: Full backtest result dict for PCA (with_costs, without_costs sub-dicts).
        ae_results: Full backtest result dict for AE.
        pca_metrics: Metrics dict for PCA (with_costs, without_costs).
        ae_metrics: Metrics dict for AE.
        save_path: If given, save figure to this path (recommend high DPI).

    Returns:
        Matplotlib Figure.
    """
    pca_wc = pca_results["with_costs"]["daily_returns"]
    pca_nc = pca_results["without_costs"]["daily_returns"]
    ae_wc = ae_results["with_costs"]["daily_returns"]
    ae_nc = ae_results["without_costs"]["daily_returns"]

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        "Statistical Arbitrage: PCA vs Deep Autoencoder — Full Comparison Dashboard",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)

    # (0,0) — Cumulative PnL
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(_cum(pca_wc).values, color=PCA_COLOR, linewidth=2, label="PCA (with costs)")
    ax0.plot(_cum(ae_wc).values, color=AE_COLOR, linewidth=2, label="AE (with costs)")
    ax0.plot(_cum(pca_nc).values, color=PCA_COLOR, linewidth=1, linestyle="--", label="PCA (no costs)")
    ax0.plot(_cum(ae_nc).values, color=AE_COLOR, linewidth=1, linestyle="--", label="AE (no costs)")
    ax0.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax0.set_title("Cumulative PnL", fontsize=11)
    ax0.legend(fontsize=8, ncol=4)
    ax0.set_ylabel("Cumulative Return")

    # (1,0) — Drawdown PCA
    ax1 = fig.add_subplot(gs[1, 0])
    dd_pca = _dd(pca_wc)
    ax1.fill_between(range(len(dd_pca)), dd_pca.values, 0, alpha=0.65, color=PCA_COLOR)
    ax1.set_title("PCA Drawdown", fontsize=10)
    ax1.set_ylabel("Drawdown")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # (1,1) — Drawdown AE
    ax2 = fig.add_subplot(gs[1, 1])
    dd_ae = _dd(ae_wc)
    ax2.fill_between(range(len(dd_ae)), dd_ae.values, 0, alpha=0.65, color=AE_COLOR)
    ax2.set_title("AE Drawdown", fontsize=10)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # (2,0) — Rolling Sharpe comparison
    ax3 = fig.add_subplot(gs[2, 0])
    W = 63
    pca_rs = (pca_wc.rolling(W).mean() / pca_wc.rolling(W).std(ddof=1) * np.sqrt(252)).fillna(0)
    ae_rs = (ae_wc.rolling(W).mean() / ae_wc.rolling(W).std(ddof=1) * np.sqrt(252)).fillna(0)
    ax3.plot(range(len(pca_rs)), pca_rs.values, color=PCA_COLOR, linewidth=1.5, label="PCA")
    ax3.plot(range(len(ae_rs)), ae_rs.values, color=AE_COLOR, linewidth=1.5, label="AE")
    ax3.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax3.set_title(f"Rolling {W}-Day Sharpe", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.set_ylabel("Sharpe")

    # (2,1) — Metrics bar chart
    ax4 = fig.add_subplot(gs[2, 1])
    metric_names = ["sharpe_ratio", "hit_ratio", "annualized_return"]
    metric_labels = ["Sharpe", "Hit Ratio", "Ann. Return"]
    x = np.arange(len(metric_names))
    pca_vals = [pca_metrics["with_costs"][m] for m in metric_names]
    ae_vals = [ae_metrics["with_costs"][m] for m in metric_names]
    ax4.bar(x - 0.2, pca_vals, 0.4, color=PCA_COLOR, alpha=0.85, label="PCA")
    ax4.bar(x + 0.2, ae_vals, 0.4, color=AE_COLOR, alpha=0.85, label="AE")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_labels, fontsize=9)
    ax4.axhline(0, color="black", linewidth=0.7)
    ax4.set_title("Key Metrics (with costs)", fontsize=10)
    ax4.legend(fontsize=8)

    # (3,0) — Transaction costs PCA
    ax5 = fig.add_subplot(gs[3, 0])
    tc_pca = pca_results["with_costs"]["transaction_costs"] * 100
    ax5.bar(range(len(tc_pca)), tc_pca.values, color=PCA_COLOR, alpha=0.5, width=1)
    ax5.set_title("PCA Daily Transaction Costs (%)", fontsize=10)
    ax5.set_ylabel("Cost (%)")

    # (3,1) — Transaction costs AE
    ax6 = fig.add_subplot(gs[3, 1])
    tc_ae = ae_results["with_costs"]["transaction_costs"] * 100
    ax6.bar(range(len(tc_ae)), tc_ae.values, color=AE_COLOR, alpha=0.5, width=1)
    ax6.set_title("AE Daily Transaction Costs (%)", fontsize=10)
    ax6.set_ylabel("Cost (%)")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
