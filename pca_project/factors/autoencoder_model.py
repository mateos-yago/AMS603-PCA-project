"""Deep autoencoder factor model implemented in PyTorch."""

from __future__ import annotations

import copy
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pca_project.factors.base_factor_model import BaseFactorModel

logger = logging.getLogger(__name__)

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "relu": nn.ReLU,
}


def _log_space_sizes(n_assets: int, bottleneck: int, depth: int) -> list[int]:
    """Compute encoder hidden-layer output sizes using log-linear spacing.

    Args:
        n_assets: Input/output dimension.
        bottleneck: Bottleneck (latent) dimension.
        depth: Number of hidden layers on each side of the bottleneck.

    Returns:
        List of ``depth`` integers: the output size of each encoder hidden layer,
        ending at ``bottleneck``.
    """
    if depth == 0:
        return [bottleneck]
    # depth+1 linearly-spaced values on a log scale from n_assets → bottleneck
    log_sizes = np.linspace(np.log(n_assets), np.log(max(bottleneck, 1)), depth + 2)
    sizes = [max(bottleneck, int(round(np.exp(s)))) for s in log_sizes[1:]]
    sizes[-1] = bottleneck  # guarantee exact bottleneck
    return sizes


class Autoencoder(nn.Module):
    """Symmetric fully-connected autoencoder with geometric layer sizing.

    The encoder has ``depth`` hidden layers whose sizes decrease geometrically
    (log-linear) from ``n_assets`` to ``bottleneck``. The decoder mirrors the
    encoder in reverse. No activation is applied after the bottleneck or the
    final reconstruction layer (linear output to allow negative values).

    Args:
        n_assets: Dimensionality of the input (number of stocks).
        bottleneck: Size of the latent bottleneck layer.
        depth: Number of hidden layers on each side of the bottleneck.
        activation: Activation function name — one of "tanh", "elu", "relu".
        dropout_rate: Dropout probability applied after each activation.
    """

    def __init__(
        self,
        n_assets: int,
        bottleneck: int,
        depth: int,
        activation: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}."
            )
        act_cls = _ACTIVATIONS[activation]

        layer_sizes = _log_space_sizes(n_assets, bottleneck, depth)
        # layer_sizes is a list of length depth+1: [h1, h2, ..., bottleneck]
        # Full encoder sizes: n_assets → h1 → h2 → ... → bottleneck
        enc_in = [n_assets] + layer_sizes[:-1]
        enc_out = layer_sizes

        encoder_layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(enc_in, enc_out)):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(enc_in) - 1:  # no activation/dropout at bottleneck
                encoder_layers.append(act_cls())
                if dropout_rate > 0:
                    encoder_layers.append(nn.Dropout(dropout_rate))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: mirrors encoder
        dec_in = layer_sizes[::-1]
        dec_out = layer_sizes[::-1][1:] + [n_assets]
        decoder_layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dec_in, dec_out)):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dec_in) - 1:  # no activation/dropout at final output
                decoder_layers.append(act_cls())
                if dropout_rate > 0:
                    decoder_layers.append(nn.Dropout(dropout_rate))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and decoder.

        Args:
            x: Input tensor of shape ``(batch, n_assets)``.

        Returns:
            Tuple of ``(reconstruction, bottleneck_representation)``.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the bottleneck representation without decoding.

        Args:
            x: Input tensor of shape ``(batch, n_assets)``.

        Returns:
            Bottleneck tensor of shape ``(batch, bottleneck)``.
        """
        return self.encoder(x)


class AutoencoderModel(BaseFactorModel):
    """BaseFactorModel wrapper around the PyTorch Autoencoder.

    Handles training with early stopping, residual extraction, and
    persistence (save/load). Uses MSE loss and Adam optimizer.

    Args:
        config: Project configuration dict.
        bottleneck: Latent dimension. Defaults to ``config['autoencoder']['default_bottleneck']``.
        depth: Number of hidden layers per side. Defaults to ``config['autoencoder']['default_depth']``.
        activation: Activation function name. Defaults to ``config['autoencoder']['default_activation']``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        bottleneck: int | None = None,
        depth: int | None = None,
        activation: str | None = None,
    ) -> None:
        super().__init__(config)
        ae_cfg = config["autoencoder"]
        self.bottleneck: int = bottleneck if bottleneck is not None else ae_cfg["default_bottleneck"]
        self.depth: int = depth if depth is not None else ae_cfg["default_depth"]
        self.activation: str = activation if activation is not None else ae_cfg["default_activation"]
        self.dropout_rate: float = ae_cfg["dropout_rate"]
        self.lr: float = ae_cfg["learning_rate"]
        self.weight_decay: float = ae_cfg["weight_decay"]
        self.batch_size: int = ae_cfg["batch_size"]
        self.max_epochs: int = ae_cfg["max_epochs"]
        self.patience: int = ae_cfg["early_stopping_patience"]

        # Fitted attributes
        self.network_: Autoencoder | None = None
        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []
        self.best_epoch_: int | None = None
        self.final_train_loss_: float | None = None
        self.final_val_loss_: float | None = None
        self.n_assets_: int | None = None
        self.tickers_: list[str] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, returns: pd.DataFrame, val_returns: pd.DataFrame) -> dict:
        """Train the autoencoder with early stopping on the validation set.

        Args:
            returns: Training returns ``(T_train, N)``.
            val_returns: Validation returns ``(T_val, N)``.

        Returns:
            Training history dict with keys: train_losses, val_losses,
            best_epoch, stopped_early.
        """
        # Reproducibility: seed before weight initialization
        torch.manual_seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

        T, N = returns.shape
        self.n_assets_ = N
        self.tickers_ = list(returns.columns)

        logger.info(
            "Training AutoencoderModel: N=%d, bottleneck=%d, depth=%d, act=%s",
            N, self.bottleneck, self.depth, self.activation,
        )

        X_train = torch.FloatTensor(returns.values)
        X_val = torch.FloatTensor(val_returns.values)

        train_loader = DataLoader(
            TensorDataset(X_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.network_ = Autoencoder(N, self.bottleneck, self.depth, self.activation, self.dropout_rate)
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_weights = None
        no_improve_count = 0
        self.train_losses_ = []
        self.val_losses_ = []

        pbar = tqdm(range(self.max_epochs), desc="AE training", leave=False)
        for epoch in pbar:
            # --- Training pass ---
            self.network_.train()
            batch_losses: list[float] = []
            for (x_batch,) in train_loader:
                optimizer.zero_grad()
                x_hat, _ = self.network_(x_batch)
                loss = criterion(x_hat, x_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_loss = float(np.mean(batch_losses))

            # --- Validation pass ---
            self.network_.eval()
            with torch.no_grad():
                x_hat_val, _ = self.network_(X_val)
                val_loss = criterion(x_hat_val, X_val).item()

            self.train_losses_.append(train_loss)
            self.val_losses_.append(val_loss)
            pbar.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"})

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.network_.state_dict())
                self.best_epoch_ = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d). Best val loss=%.6f",
                        epoch, self.patience, best_val_loss,
                    )
                    break

        # Restore best weights
        if best_weights is not None:
            self.network_.load_state_dict(best_weights)

        self.final_train_loss_ = self.train_losses_[self.best_epoch_]
        self.final_val_loss_ = best_val_loss
        self._is_fitted = True

        stopped_early = no_improve_count >= self.patience
        logger.info(
            "AE training done. Best epoch=%d, val_loss=%.6f, stopped_early=%s",
            self.best_epoch_, self.final_val_loss_, stopped_early,
        )
        return {
            "train_losses": self.train_losses_,
            "val_losses": self.val_losses_,
            "best_epoch": self.best_epoch_,
            "stopped_early": stopped_early,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _to_tensor_and_reconstruct(
        self, returns: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run returns through the network in eval mode.

        Returns:
            Tuple of ``(actual_np, reconstruction_np)``, both shape ``(T, N)``.
        """
        self.validate_not_fitted()
        self.network_.eval()
        X = torch.FloatTensor(returns.values)
        with torch.no_grad():
            x_hat, _ = self.network_(X)
        return returns.values, x_hat.numpy()

    def get_factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return the autoencoder reconstruction (systematic component).

        Args:
            returns: Log-return DataFrame ``(T, N)``.

        Returns:
            Reconstructed return DataFrame ``(T, N)``.
        """
        _, reconstruction = self._to_tensor_and_reconstruct(returns)
        return pd.DataFrame(reconstruction, index=returns.index, columns=returns.columns)

    def get_residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute idiosyncratic residuals: actual − reconstruction.

        Args:
            returns: Log-return DataFrame ``(T, N)``.

        Returns:
            Residual DataFrame ``(T, N)``.
        """
        actual, reconstruction = self._to_tensor_and_reconstruct(returns)
        residuals = actual - reconstruction
        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)

    def get_bottleneck_representation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Extract the latent bottleneck representation.

        Args:
            returns: Log-return DataFrame ``(T, N)``.

        Returns:
            Bottleneck DataFrame ``(T, bottleneck)`` with columns factor_0..factor_k.
        """
        self.validate_not_fitted()
        self.network_.eval()
        X = torch.FloatTensor(returns.values)
        with torch.no_grad():
            z = self.network_.encode(X)
        cols = [f"factor_{j}" for j in range(self.bottleneck)]
        return pd.DataFrame(z.numpy(), index=returns.index, columns=cols)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the fitted model to disk using pickle.

        Args:
            path: File path (should end in ``.pkl``).
        """
        self.validate_not_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("AutoencoderModel saved to %s", path)

    @classmethod
    def load(cls, path: str, config: dict[str, Any]) -> "AutoencoderModel":
        """Load a previously saved AutoencoderModel from disk.

        Args:
            path: File path of the pickled model.
            config: Project configuration dict (used to reconstruct if needed).

        Returns:
            Loaded ``AutoencoderModel`` instance.
        """
        with open(path, "rb") as fh:
            model: AutoencoderModel = pickle.load(fh)
        logger.info("AutoencoderModel loaded from %s", path)
        return model
