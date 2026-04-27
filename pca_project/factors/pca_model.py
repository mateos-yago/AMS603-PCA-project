"""PCA factor model following the Avellaneda & Lee (2010) procedure."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from pca_project.factors.base_factor_model import BaseFactorModel

logger = logging.getLogger(__name__)


class PCAModel(BaseFactorModel):
    """PCA-based factor model using eigenportfolio decomposition.

    Implements the Avellaneda & Lee (2010) statistical arbitrage procedure:
      1. Compute the empirical correlation matrix of standardized returns.
      2. Extract the top-k eigenvectors (eigenportfolios).
      3. Project each stock's returns onto the eigenportfolios to obtain
         factor loadings (betas) via OLS.
      4. Residuals = actual returns − beta-weighted factor returns.

    Residuals represent the idiosyncratic component of each stock's return,
    which is modelled as an OU process for signal generation.

    Args:
        config: Project configuration dict.
        n_factors: Number of PCA factors (top eigenvectors) to retain.
                   Defaults to ``config['pca']['default_n_factors']``.
    """

    def __init__(self, config: dict[str, Any], n_factors: int | None = None) -> None:
        super().__init__(config)
        self.n_factors: int = n_factors if n_factors is not None else config["pca"]["default_n_factors"]

        # Fitted attributes — populated by fit()
        self.eigenvalues_: np.ndarray | None = None
        self.eigenvectors_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_variance_explained_: float | None = None
        self.eigenportfolio_weights_: np.ndarray | None = None
        self.betas_: np.ndarray | None = None
        self.factor_returns_train_: pd.DataFrame | None = None
        self.correlation_matrix_: np.ndarray | None = None
        self.per_stock_mean_: np.ndarray | None = None
        self.per_stock_std_: np.ndarray | None = None
        self.tickers_: list[str] | None = None
        self.n_assets_: int | None = None
        self.n_factors_: int | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: pd.DataFrame) -> None:
        """Fit the PCA factor model on training returns.

        Follows the Avellaneda & Lee (2010) procedure:
          Step 1 — Compute per-stock time-series mean and std; standardize returns.
          Step 2 — Compute empirical correlation matrix.
          Step 3 — Eigendecomposition; keep top-k eigenvectors.
          Step 4 — Compute eigenportfolio weights Q = v / sigma_i.
          Step 5 — OLS regression of each stock on factor returns to get betas.

        Args:
            returns: Standardized log-return DataFrame ``(T_train, N)``.
                     Input should already be cross-sectionally standardized.
        """
        T, N = returns.shape
        self.tickers_ = list(returns.columns)
        self.n_assets_ = N
        self.n_factors_ = self.n_factors

        logger.info(
            "Fitting PCAModel: T=%d days, N=%d assets, k=%d factors", T, N, self.n_factors
        )

        R = returns.values  # (T, N)

        # Step 1 — Per-stock time-series mean and std (for eigenportfolio normalization)
        self.per_stock_mean_ = R.mean(axis=0)        # (N,)
        self.per_stock_std_ = R.std(axis=0, ddof=1)  # (N,)
        # Guard against zero-std stocks (should not happen after cross-sect. std, but be safe)
        self.per_stock_std_ = np.where(self.per_stock_std_ == 0, 1.0, self.per_stock_std_)

        Y = (R - self.per_stock_mean_) / self.per_stock_std_  # (T, N)

        # Step 2 — Empirical correlation matrix (N, N)
        self.correlation_matrix_ = (Y.T @ Y) / (T - 1)

        # Step 3 — Eigendecomposition
        # eigh returns eigenvalues in ascending order → reverse for descending
        eigenvalues_all, eigenvectors_all = np.linalg.eigh(self.correlation_matrix_)
        idx = np.argsort(eigenvalues_all)[::-1]
        eigenvalues_all = eigenvalues_all[idx]
        eigenvectors_all = eigenvectors_all[:, idx]  # (N, N) each column is an eigenvector

        k = self.n_factors
        self.eigenvalues_ = eigenvalues_all[:k]
        self.eigenvectors_ = eigenvectors_all[:, :k]  # (N, k)

        total_variance = eigenvalues_all.sum()
        self.explained_variance_ratio_ = self.eigenvalues_ / total_variance
        self.cumulative_variance_explained_ = float(self.explained_variance_ratio_.sum())

        logger.info(
            "PCA: top-%d factors explain %.1f%% of total variance",
            k,
            self.cumulative_variance_explained_ * 100,
        )

        # Step 4 — Eigenportfolio weights: Q^(j)_i = v^(j)_i / sigma_i
        # Shape (N, k): each column j is the weight vector for eigenportfolio j
        self.eigenportfolio_weights_ = self.eigenvectors_ / self.per_stock_std_[:, np.newaxis]

        # Step 5 — Factor returns on training data: F_{j,t} = sum_i Q^(j)_i * R_{i,t}
        factor_returns_np = R @ self.eigenportfolio_weights_  # (T, k)
        self.factor_returns_train_ = pd.DataFrame(
            factor_returns_np,
            index=returns.index,
            columns=[f"factor_{j}" for j in range(k)],
        )

        # Step 6 — OLS regression: R_{i,t} = beta_{i,0} + sum_j beta_{i,j} F_{j,t} + eps
        # Design matrix: (T, k+1) with intercept
        X_design = np.column_stack([np.ones(T), factor_returns_np])  # (T, k+1)
        # Solve for all N stocks simultaneously: (k+1, N)
        betas_T, _, _, _ = np.linalg.lstsq(X_design, R, rcond=None)
        self.betas_ = betas_T.T  # (N, k+1): row i = [beta_{i,0}, beta_{i,1}, ..., beta_{i,k}]

        self._is_fitted = True
        logger.info("PCAModel fitting complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute factor returns for held-out data using fitted eigenportfolio weights.

        Args:
            returns: Log-return DataFrame ``(T, N)``.

        Returns:
            Factor return DataFrame ``(T, k)`` where k = n_factors.
        """
        self.validate_not_fitted()
        R = returns.values  # (T, N)
        factor_returns_np = R @ self.eigenportfolio_weights_  # (T, k)
        return pd.DataFrame(
            factor_returns_np,
            index=returns.index,
            columns=[f"factor_{j}" for j in range(self.n_factors_)],
        )

    def get_residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute idiosyncratic residuals for held-out returns.

        For each stock i:
          R_hat_{i,t} = beta_{i,0} + sum_j beta_{i,j} * F_{j,t}
          epsilon_{i,t} = R_{i,t} - R_hat_{i,t}

        Args:
            returns: Log-return DataFrame ``(T, N)``.

        Returns:
            Residual DataFrame of the same shape ``(T, N)``.
        """
        self.validate_not_fitted()
        factor_returns = self.get_factor_returns(returns)  # (T, k)
        F = factor_returns.values                           # (T, k)
        R = returns.values                                  # (T, N)

        # Systematic component: R_hat = [1, F] @ betas.T
        X_design = np.column_stack([np.ones(len(F)), F])   # (T, k+1)
        R_hat = X_design @ self.betas_.T                   # (T, N)

        residuals = R - R_hat
        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_variance_explained_summary(self) -> dict:
        """Return a summary of variance explained by the fitted factors.

        Returns:
            Dict with keys: n_factors, eigenvalues, explained_variance_ratio,
            cumulative_variance_explained.
        """
        self.validate_not_fitted()
        return {
            "n_factors": self.n_factors_,
            "eigenvalues": self.eigenvalues_.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "cumulative_variance_explained": self.cumulative_variance_explained_,
        }
