"""Visualization functions for signal and residual diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
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


def plot_zscore_timeseries(
    zscores: pd.DataFrame,
    ticker: str,
    entry_threshold: float,
    exit_threshold: float,
    signals: pd.DataFrame,
    save_path: str | None = None,
) -> Figure:
    """Z-score time series for a single stock with position background shading.

    Args:
        zscores: Z-score DataFrame ``(T, N)``.
        ticker: Column to plot.
        entry_threshold: Entry threshold (horizontal dashed lines at ±value).
        exit_threshold: Exit threshold (horizontal dotted lines at ±value).
        signals: Signal DataFrame ``(T, N)`` with values in ``{-1, 0, +1}``.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    z = zscores[ticker].dropna()
    sig = signals[ticker].reindex(z.index).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Shade position regions
    for i in range(len(z)):
        s = sig.iloc[i]
        if s == 1:
            ax.axvspan(z.index[i], z.index[min(i + 1, len(z) - 1)], alpha=0.12, color="green")
        elif s == -1:
            ax.axvspan(z.index[i], z.index[min(i + 1, len(z) - 1)], alpha=0.12, color="red")

    ax.plot(z.index, z.values, color="black", linewidth=1, label="Z-score")
    ax.axhline(entry_threshold, color="blue", linestyle="--", linewidth=1, label=f"Entry ±{entry_threshold}")
    ax.axhline(-entry_threshold, color="blue", linestyle="--", linewidth=1)
    ax.axhline(exit_threshold, color="grey", linestyle=":", linewidth=1, label=f"Exit ±{exit_threshold}")
    ax.axhline(-exit_threshold, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Z-score (s-score)")
    ax.set_title(f"Z-Score Time Series — {ticker}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_residual_acf(
    residuals: pd.DataFrame,
    ticker: str,
    n_lags: int = 40,
    save_path: str | None = None,
) -> Figure:
    """Autocorrelation function of the residual series for one stock.

    Args:
        residuals: Residual DataFrame ``(T, N)``.
        ticker: Column to analyse.
        n_lags: Number of lags to display.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    series = residuals[ticker].dropna().values
    T = len(series)
    acf_vals = [1.0]
    mean = series.mean()
    var = ((series - mean) ** 2).sum()
    for lag in range(1, n_lags + 1):
        cov = ((series[lag:] - mean) * (series[:-lag] - mean)).sum()
        acf_vals.append(cov / var if var > 0 else 0.0)

    conf = 1.96 / np.sqrt(T)
    lags = list(range(n_lags + 1))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(lags, acf_vals, color=PCA_COLOR, alpha=0.7, width=0.6)
    ax.axhspan(-conf, conf, alpha=0.15, color="blue", label="95% CI")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Residual ACF — {ticker}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_ou_parameter_distribution(
    ou_params_df: pd.DataFrame,
    param: str = "kappa",
    min_kappa: float | None = None,
    save_path: str | None = None,
) -> Figure:
    """Histogram of an OU parameter across all stocks.

    Args:
        ou_params_df: DataFrame ``(T, N)`` of a single OU parameter (e.g. kappa).
        param: Parameter name (for axis label).
        min_kappa: If provided, draw a vertical line at this threshold.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    vals = ou_params_df.values.ravel()
    vals = vals[~np.isnan(vals)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals, bins=60, color=PCA_COLOR, alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(vals), color="darkorange", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(vals):.1f}")
    if min_kappa is not None:
        ax.axvline(min_kappa, color="red", linestyle="--", linewidth=1.5, label=f"min_kappa = {min_kappa}")
    ax.set_xlabel(param)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of OU Parameter: {param}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_signal_heatmap(
    signals: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Heatmap of long/short positions over time.

    Args:
        signals: Signal DataFrame ``(T, N)`` with values in ``{-1, 0, +1}``.
        start_date: Optional start date for subsetting.
        end_date: Optional end date for subsetting.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    sig = signals
    if start_date:
        sig = sig.loc[start_date:]
    if end_date:
        sig = sig.loc[:end_date]

    # Select top 50 most active stocks
    activity = (sig != 0).sum()
    top_cols = activity.nlargest(min(50, len(activity))).index
    sig = sig[top_cols].T  # (stocks, time) for heatmap

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(["#E84855", "white", "#2E86AB"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, max(6, len(top_cols) * 0.18)))
    ax.imshow(sig.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # x-axis: sampled dates
    n_ticks = min(10, sig.shape[1])
    tick_locs = np.linspace(0, sig.shape[1] - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([str(sig.columns[i].date()) for i in tick_locs], rotation=45, fontsize=7)
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols, fontsize=6)
    ax.set_title("Signal Map — Long/Short Positions Over Time")
    from matplotlib.patches import Patch
    legend = [Patch(color="#2E86AB", label="Long"), Patch(color="#E84855", label="Short"), Patch(color="white", label="Flat")]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_position_counts(
    n_long: pd.Series,
    n_short: pd.Series,
    save_path: str | None = None,
) -> Figure:
    """Stacked area chart of active long and short position counts.

    Args:
        n_long: Series of daily long position counts.
        n_short: Series of daily short position counts.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(n_long.index, n_long.values, alpha=0.7, color="#2E86AB", label="Long positions")
    ax.fill_between(n_short.index, -n_short.values, alpha=0.7, color="#E84855", label="Short positions (reflected)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Number of Positions")
    ax.set_title("Number of Active Long and Short Positions")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)
