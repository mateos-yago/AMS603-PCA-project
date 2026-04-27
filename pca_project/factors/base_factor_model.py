"""Abstract base class for all factor models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFactorModel(ABC):
    """Contract shared by PCAModel and AutoencoderModel.

    Both models must be fitted on training data and then used to decompose
    held-out returns into a systematic (factor) component and idiosyncratic
    residuals. The residuals feed into the OU process and signal generator.

    Args:
        config: Project configuration dict from ``load_config()``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> None:
        """Fit the factor model on training returns.

        Args:
            returns: Standardized log-return DataFrame of shape ``(T_train, N)``.
        """

    @abstractmethod
    def get_residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute idiosyncratic residuals for the given returns.

        Args:
            returns: Log-return DataFrame of shape ``(T, N)``.

        Returns:
            Residual DataFrame of the same shape ``(T, N)``.
        """

    @abstractmethod
    def get_factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return the systematic (reconstructed) component of returns.

        Args:
            returns: Log-return DataFrame of shape ``(T, N)``.

        Returns:
            Reconstructed returns DataFrame of the same shape ``(T, N)``.
        """

    def validate_not_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call fit() first."
            )
