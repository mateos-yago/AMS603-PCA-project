"""Visualization functions for backtest performance analysis."""

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


def _cumulative(daily_returns: pd.Series) -> pd.Series:
    return (1.0 + daily_returns).cumprod()


def _drawdown(daily_returns: pd.Series) -> pd.Series:
    wealth = _cumulative(daily_returns)
    rolling_max = wealth.cummax()
    return (wealth - rolling_max) / rolling_max


def plot_cumulative_pnl(
    daily_returns_with_costs: pd.Series,
    daily_returns_no_costs: pd.Series,
    label: str,
    save_path: str | None = None,
) -> Figure:
    """Cumulative PnL with and without transaction costs, gap shaded.

    Args:
        daily_returns_with_costs: Daily returns after transaction costs.
        daily_returns_no_costs: Daily returns before transaction costs.
        label: Strategy label used in title.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    cum_wc = _cumulative(daily_returns_with_costs)
    cum_nc = _cumulative(daily_returns_no_costs)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(cum_nc.index, cum_nc.values, color=PCA_COLOR, linewidth=1.8, linestyle="--", label="Without costs")
    ax.plot(cum_wc.index, cum_wc.values, color=PCA_COLOR, linewidth=2.0, label="With costs")
    ax.fill_between(cum_nc.index, cum_wc.values, cum_nc.values, alpha=0.15, color="grey", label="Cost drag")
    ax.annotate(f"{cum_wc.iloc[-1]:.3f}", xy=(cum_wc.index[-1], cum_wc.iloc[-1]),
                xytext=(10, 0), textcoords="offset points", fontsize=9, color=PCA_COLOR)
    ax.annotate(f"{cum_nc.iloc[-1]:.3f}", xy=(cum_nc.index[-1], cum_nc.iloc[-1]),
                xytext=(10, 5), textcoords="offset points", fontsize=9, color="grey")
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Cumulative Return (indexed to 1)")
    ax.set_title(f"Cumulative PnL — {label}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_drawdown(
    daily_returns: pd.Series,
    label: str,
    save_path: str | None = None,
) -> Figure:
    """Drawdown area chart with maximum drawdown annotated.

    Args:
        daily_returns: Daily return series.
        label: Strategy label.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    dd = _drawdown(daily_returns)
    max_dd_idx = dd.idxmin()
    max_dd_val = dd.min()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(dd.index, dd.values, 0, alpha=0.65, color=AE_COLOR, label="Drawdown")
    ax.axvline(max_dd_idx, color="black", linewidth=1, linestyle="--")
    ax.annotate(
        f"Max DD: {max_dd_val:.1%}",
        xy=(max_dd_idx, max_dd_val),
        xytext=(20, -15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )
    ax.set_ylabel("Drawdown")
    ax.set_title(f"Drawdown — {label}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_rolling_sharpe(
    daily_returns: pd.Series,
    label: str,
    window: int = 63,
    save_path: str | None = None,
) -> Figure:
    """Rolling annualized Sharpe ratio.

    Args:
        daily_returns: Daily return series.
        label: Strategy label.
        window: Rolling window in trading days (default 63 ≈ 1 quarter).
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std(ddof=1)
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=PCA_COLOR, linewidth=1.5, label=f"{window}-day Sharpe")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color="green", linewidth=0.8, linestyle=":", label="Sharpe = 1")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=(rolling_sharpe.values >= 0), alpha=0.2, color="green")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=(rolling_sharpe.values < 0), alpha=0.2, color="red")
    ax.set_ylabel("Rolling Sharpe Ratio (annualised)")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio — {label}")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_monthly_returns_heatmap(
    daily_returns: pd.Series,
    label: str,
    save_path: str | None = None,
) -> Figure:
    """Calendar heatmap of monthly returns.

    Args:
        daily_returns: Daily return series with DatetimeIndex.
        label: Strategy label.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).mean().unstack()
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.6)))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7,
                        color="black" if abs(val) < vmax * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Monthly Return (%)")
    ax.set_title(f"Monthly Returns — {label}")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_transaction_costs_impact(
    daily_returns_with_costs: pd.Series,
    daily_returns_no_costs: pd.Series,
    transaction_costs: pd.Series,
    label: str,
    save_path: str | None = None,
) -> Figure:
    """Three-panel transaction cost analysis figure.

    Args:
        daily_returns_with_costs: Returns after costs.
        daily_returns_no_costs: Returns before costs.
        transaction_costs: Daily cost series as fraction of portfolio.
        label: Strategy label.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    cum_wc = _cumulative(daily_returns_with_costs)
    cum_nc = _cumulative(daily_returns_no_costs)
    tc_pct = transaction_costs * 100

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    ax1.plot(cum_nc.index, cum_nc.values, color=PCA_COLOR, linestyle="--", linewidth=1.5, label="Without costs")
    ax1.plot(cum_wc.index, cum_wc.values, color=PCA_COLOR, linewidth=2, label="With costs")
    ax1.fill_between(cum_nc.index, cum_wc.values, cum_nc.values, alpha=0.15, color="grey")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(fontsize=9)
    ax1.set_title(f"Transaction Cost Analysis — {label}")

    ax2.bar(tc_pct.index, tc_pct.values, color=AE_COLOR, alpha=0.6, width=1)
    ax2.set_ylabel("Daily Cost (%)")

    rolling_cost = tc_pct.rolling(63).sum()
    ax3.plot(rolling_cost.index, rolling_cost.values, color="darkorange", linewidth=1.5)
    ax3.fill_between(rolling_cost.index, rolling_cost.values, 0, alpha=0.2, color="darkorange")
    ax3.set_ylabel("Rolling 63-Day Cost (%)")
    ax3.set_xlabel("Date")

    fig.tight_layout()
    return _save(fig, save_path)
