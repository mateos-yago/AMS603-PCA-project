"""Ornstein-Uhlenbeck parameter estimation, Z-score generation, and signal generation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OUProcess:
    """Estimate OU parameters from a 1-D residual series.

    Follows the discrete-time AR(1) mapping from Avellaneda & Lee (2010),
    Appendix A.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.ou_lookback_days: int = config["signals"]["ou_lookback_days"]
        self.min_kappa: float = config["signals"]["min_kappa"]

    def estimate_parameters(self, residuals: np.ndarray) -> dict:
        """Estimate OU parameters from a 1-D array of residual returns.

        Step 1: Cumulative residual process X_k = cumsum(residuals).
        Step 2: AR(1) regression X_{n+1} = a + b*X_n + noise.
        Step 3: Map to continuous-time OU parameters.
        Step 4: Validity check (kappa >= min_kappa and b in (0, 1)).

        Args:
            residuals: 1-D array of idiosyncratic residual returns over
                       the lookback window.

        Returns:
            Dict with keys: a, b, kappa, m, sigma, sigma_eq, var_noise, is_valid.
        """
        invalid = {
            "a": np.nan, "b": np.nan, "kappa": np.nan, "m": np.nan,
            "sigma": np.nan, "sigma_eq": np.nan, "var_noise": np.nan,
            "is_valid": False,
        }

        if len(residuals) < 4:
            return invalid

        X = np.cumsum(residuals)  # X_k for k=1..T

        X_n = X[:-1]              # X_n
        X_np1 = X[1:]             # X_{n+1}

        # AR(1) OLS: X_{n+1} = a + b*X_n
        A = np.column_stack([np.ones(len(X_n)), X_n])
        try:
            result, residuals_ar, _, _ = np.linalg.lstsq(A, X_np1, rcond=None)
        except np.linalg.LinAlgError:
            return invalid

        a, b = result

        # Variance of AR regression noise
        X_np1_hat = a + b * X_n
        ar_residuals = X_np1 - X_np1_hat
        var_noise = float(np.var(ar_residuals, ddof=2)) if len(ar_residuals) > 2 else np.nan

        if var_noise is np.nan or var_noise <= 0:
            return invalid

        # Validity: unit root or oscillating
        if b >= 1.0 or b <= 0.0:
            return {**invalid, "a": a, "b": b}

        # Continuous-time OU parameter mapping (Avellaneda & Lee Appendix A)
        kappa = -np.log(b) * 252          # annualised mean-reversion speed
        m = a / (1.0 - b)                 # long-run mean
        sigma_eq = np.sqrt(var_noise / (1.0 - b ** 2))
        sigma = np.sqrt(var_noise * 2.0 * kappa / (1.0 - b ** 2))

        is_valid = kappa >= self.min_kappa

        return {
            "a": float(a),
            "b": float(b),
            "kappa": float(kappa),
            "m": float(m),
            "sigma": float(sigma),
            "sigma_eq": float(sigma_eq),
            "var_noise": float(var_noise),
            "is_valid": bool(is_valid),
        }


class ZScoreGenerator:
    """Compute rolling OU-based Z-scores (s-scores) for each stock.

    For each day t, estimates OU parameters from the lookback window of
    residuals and computes the centred s-score following Avellaneda & Lee.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.ou = OUProcess(config)
        self.lookback: int = config["signals"]["ou_lookback_days"]

    def compute_zscores(
        self, residuals: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute daily rolling Z-scores from idiosyncratic residuals.

        For each day t >= lookback:
          - For each stock i, fit OU on residuals[t-lookback:t].
          - s_{i,t} = -m_i / sigma_eq_i  (X_T=0 by OLS construction).
          - Centred version: subtract cross-sectional mean of -m_j/sigma_eq_j.
          - NaN if OU fit is invalid (slow mean-reversion).

        Args:
            residuals: Residual DataFrame ``(T, N)``.

        Returns:
            Tuple of:
              - zscores: DataFrame ``(T, N)`` — Z-score per stock per day.
              - kappas:  DataFrame ``(T, N)`` — OU kappa per stock per day.
        """
        T, N = residuals.shape
        residuals_np = residuals.values  # (T, N)

        zscores_np = np.full((T, N), np.nan)
        kappas_np = np.full((T, N), np.nan)

        for t in range(self.lookback, T):
            window = residuals_np[t - self.lookback: t, :]  # (lookback, N)

            s_raw = np.full(N, np.nan)
            for i in range(N):
                window_i = window[:, i]
                x_end = np.sum(window_i)  # X_T = cumulative residual at end of window
                params = self.ou.estimate_parameters(window_i)
                if params["is_valid"]:
                    m_i = params["m"]
                    sig_eq_i = params["sigma_eq"]
                    if sig_eq_i > 0:
                        s_raw[i] = (x_end - m_i) / sig_eq_i  # A&L: s = (X_T - m) / sigma_eq
                    kappas_np[t, i] = params["kappa"]

            # Cross-sectional demeaning — removes market-wide OU drift not captured by factors
            valid_mask = ~np.isnan(s_raw)
            if valid_mask.sum() > 0:
                cs_mean = np.nanmean(s_raw)
                zscores_np[t, valid_mask] = s_raw[valid_mask] - cs_mean

        return (
            pd.DataFrame(zscores_np, index=residuals.index, columns=residuals.columns),
            pd.DataFrame(kappas_np, index=residuals.index, columns=residuals.columns),
        )


class SignalGenerator:
    """Convert Z-scores into long/short/flat position signals.

    Maintains stateful positions: signals are based on threshold crossings.
    Entry when |z| > entry_threshold; exit when |z| < exit_threshold.

    The signal on day t (based on returns up to day t) must be applied to
    day t+1 returns in the backtesting engine to prevent look-ahead bias.

    Args:
        config: Project configuration dict.
        zscore_entry: Entry threshold. Defaults to ``config['signals']['default_zscore_entry']``.
        zscore_exit: Exit threshold. Defaults to ``config['signals']['default_zscore_exit']``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        zscore_entry: float | None = None,
        zscore_exit: float | None = None,
    ) -> None:
        sig_cfg = config["signals"]
        self.entry: float = zscore_entry if zscore_entry is not None else sig_cfg["default_zscore_entry"]
        self.exit: float = zscore_exit if zscore_exit is not None else sig_cfg["default_zscore_exit"]

    def generate_signals(self, zscores: pd.DataFrame) -> pd.DataFrame:
        """Convert Z-score DataFrame into position signals.

        Signal values: +1 = long, -1 = short, 0 = flat.
        Positions are updated statelessly row-by-row.

        Args:
            zscores: Z-score DataFrame ``(T, N)``.

        Returns:
            Signal DataFrame ``(T, N)`` with values in ``{-1, 0, 1}``.
        """
        T, N = zscores.shape
        zscores_np = zscores.values
        signals_np = np.zeros((T, N), dtype=np.int8)
        positions = np.zeros(N, dtype=np.int8)  # current position per stock

        entry = self.entry
        exit_ = self.exit

        for t in range(T):
            z_t = zscores_np[t]  # (N,)

            for i in range(N):
                z = z_t[i]
                pos = positions[i]

                if np.isnan(z):
                    # Invalid OU fit — close any open position
                    positions[i] = 0
                elif pos == 0:
                    if z < -entry:
                        positions[i] = 1    # open long
                    elif z > entry:
                        positions[i] = -1   # open short
                elif pos == 1:  # currently long
                    if z > -exit_:
                        positions[i] = 0    # close long (z reverted past exit)
                    if z > entry:
                        positions[i] = -1   # flip to short
                elif pos == -1:  # currently short
                    if z < exit_:
                        positions[i] = 0    # close short
                    if z < -entry:
                        positions[i] = 1    # flip to long

            signals_np[t] = positions.copy()

        return pd.DataFrame(
            signals_np.astype(int),
            index=zscores.index,
            columns=zscores.columns,
        )
