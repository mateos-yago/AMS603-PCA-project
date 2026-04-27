"""Visualization functions for factor model diagnostics."""

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


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    explained_ratios: np.ndarray,
    n_factors_selected: int,
    save_path: str | None = None,
) -> Figure:
    """Bar chart of eigenvalue spectrum with cumulative variance line.

    Args:
        eigenvalues: Array of eigenvalues (all, not just top-k).
        explained_ratios: Fraction of variance explained per component.
        n_factors_selected: Number of factors selected (highlighted).
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    n_show = min(50, len(eigenvalues))
    ratios = explained_ratios[:n_show] * 100
    cumulative = np.cumsum(explained_ratios[:n_show]) * 100

    fig, ax1 = plt.subplots(figsize=(12, 5))
    colors = [PCA_COLOR if i < n_factors_selected else "#AACFD0" for i in range(n_show)]
    ax1.bar(range(1, n_show + 1), ratios, color=colors, alpha=0.85, label="Variance explained (%)")
    ax1.axvline(n_factors_selected + 0.5, color="red", linestyle="--", linewidth=1.5, label=f"Cutoff k={n_factors_selected}")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Eigenvalue Spectrum — PCA Factor Selection")

    ax2 = ax1.twinx()
    ax2.plot(range(1, n_show + 1), cumulative, color="darkorange", linewidth=2, marker="o", markersize=3, label="Cumulative (%)")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_explained_variance_vs_k(
    k_values: list[int],
    variance_explained: list[float],
    save_path: str | None = None,
) -> Figure:
    """Cumulative variance explained vs. number of PCA factors.

    Args:
        k_values: List of factor counts tested.
        variance_explained: Cumulative variance fraction for each k.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    pct = [v * 100 for v in variance_explained]
    ax.plot(k_values, pct, color=PCA_COLOR, linewidth=2.5, marker="o", markersize=7)
    for level, ls in [(45, ":"), (55, "--"), (65, "-.")]:
        ax.axhline(level, color="grey", linestyle=ls, linewidth=1, label=f"{level}%")
    ax.set_xlabel("Number of PCA Factors (k)")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title("Cumulative Explained Variance by Number of PCA Factors")
    ax.legend(title="Threshold lines")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_eigenvector_weights(
    eigenvector: np.ndarray,
    tickers: list[str],
    component_idx: int,
    save_path: str | None = None,
) -> Figure:
    """Horizontal bar chart of eigenvector component weights.

    Args:
        eigenvector: 1-D array of weights for one principal component.
        tickers: Stock ticker labels.
        component_idx: 0-based index of the eigenvector.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    order = np.argsort(np.abs(eigenvector))[::-1][:40]
    weights = eigenvector[order]
    labels = [tickers[i] for i in order]
    colors = [PCA_COLOR if w >= 0 else AE_COLOR for w in weights]

    fig, ax = plt.subplots(figsize=(8, max(6, len(labels) * 0.25)))
    ax.barh(range(len(labels)), weights[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Weight")
    ax.set_title(f"Eigenvector {component_idx + 1} — Stock Weights")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_autoencoder_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    best_epoch: int,
    log_scale: bool = False,
    save_path: str | None = None,
) -> Figure:
    """Training and validation loss curves for the autoencoder.

    Args:
        train_losses: Per-epoch training MSE loss.
        val_losses: Per-epoch validation MSE loss.
        best_epoch: Epoch where best val loss was achieved.
        log_scale: Use log y-axis if True.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, color=PCA_COLOR, label="Train loss", linewidth=1.8)
    ax.plot(epochs, val_losses, color=AE_COLOR, label="Val loss", linewidth=1.8)
    ax.axvline(best_epoch + 1, color="green", linestyle="--", linewidth=1.5, label=f"Best epoch ({best_epoch + 1})")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_reconstruction_quality(
    actual_returns: pd.DataFrame,
    reconstructed_returns: pd.DataFrame,
    ticker: str,
    save_path: str | None = None,
) -> Figure:
    """Two-panel reconstruction quality plot for a single stock.

    Args:
        actual_returns: Actual returns DataFrame ``(T, N)``.
        reconstructed_returns: Reconstructed returns DataFrame ``(T, N)``.
        ticker: Column name to plot.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    actual = actual_returns[ticker]
    recon = reconstructed_returns[ticker]
    corr = float(np.corrcoef(actual.values, recon.values)[0, 1])
    r2 = corr ** 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    ax1.plot(actual.index, actual.values, color=PCA_COLOR, alpha=0.7, linewidth=1, label="Actual")
    ax1.plot(recon.index, recon.values, color=AE_COLOR, alpha=0.7, linewidth=1, label="Reconstructed")
    ax1.set_ylabel("Return")
    ax1.set_title(f"Reconstruction Quality — {ticker}")
    ax1.legend()

    ax2.scatter(actual.values, recon.values, alpha=0.4, s=8, color=AE_COLOR)
    mn, mx = actual.values.min(), actual.values.max()
    ax2.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y=x")
    ax2.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.90), xycoords="axes fraction", fontsize=11)
    ax2.set_xlabel("Actual return")
    ax2.set_ylabel("Reconstructed return")
    ax2.legend()
    fig.tight_layout()
    return _save(fig, save_path)
