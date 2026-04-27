# Instruction 04 — Deep Autoencoder Model

All autoencoder logic lives in `stat_arb/factors/autoencoder_model.py`. The class must conform to `BaseFactorModel`.

---

## 4.1 PyTorch Network Architecture

### Class: `Autoencoder(nn.Module)`

This is the raw PyTorch module — not the `BaseFactorModel` subclass.

```
Autoencoder(
    n_assets: int,
    bottleneck: int,
    depth: int,
    activation: str,
    dropout_rate: float
)
```

**Architecture rules**:

The hidden layer sizes are determined geometrically. Given `n_assets=N`, `bottleneck=k`, `depth=d`, and `hidden_scale=S` from config:

- The encoder has `depth` hidden layers. Layer sizes decrease from `N` toward `k`:
  - Layer sizes: `[N, N//S, N//S², ..., k]` — round to nearest integer, ensure each layer is ≥ k
  - More precisely: intermediate size at encoder layer `l` (0-indexed) = `max(k, int(N * (k/N)**(l+1)/(depth+1)... ))`

- **Simpler geometric rule** (use this): compute `d+1` linearly-spaced sizes on a log scale between `N` and `k`, rounding to int. These are the encoder layer output sizes. The decoder mirrors them in reverse.

- Example with N=400, k=15, d=2:
  - Log-spaced: `[400, ~92, ~21, 15]` (encoder outputs at each layer)
  - Full encoder: Linear(400→92) → Act → Dropout → Linear(92→21) → Act → Dropout → Linear(21→15)
  - Bottleneck: 15-dim representation (no activation after bottleneck)
  - Full decoder: Linear(15→21) → Act → Dropout → Linear(21→92) → Act → Dropout → Linear(92→400)
  - Final output layer: **no activation** (linear output — reconstructed returns can be negative)

**Activation functions**:
- Accept `activation: str` ∈ `["tanh", "elu", "relu"]`
- Map string to `nn.Tanh()`, `nn.ELU()`, `nn.ReLU()`
- Raise `ValueError` for unknown activation

**Dropout**: Apply after each activation (except at the bottleneck and output).

**Forward pass**:
```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (reconstruction, bottleneck_representation)."""
    z = self.encoder(x)
    x_hat = self.decoder(z)
    return x_hat, z
```

**Get bottleneck representation without decoding**:
```python
def encode(self, x: torch.Tensor) -> torch.Tensor:
    return self.encoder(x)
```

---

## 4.2 `AutoencoderModel(BaseFactorModel)`

```
AutoencoderModel(
    config: dict,
    bottleneck: int | None = None,
    depth: int | None = None,
    activation: str | None = None
)
```

Default each parameter from config if None.

---

### Method: `fit(self, returns: pd.DataFrame, val_returns: pd.DataFrame) -> dict`

**Training procedure**:

1. **Data preparation**:
   - Convert DataFrames to `torch.FloatTensor`
   - Shape: `(T_train, N)` and `(T_val, N)`
   - Create `TensorDataset` + `DataLoader` for training data (shuffle=True, batch_size from config)

2. **Initialize network**:
   - `self.network_ = Autoencoder(N, bottleneck, depth, activation, dropout_rate)`
   - Optimizer: `torch.optim.Adam(lr, weight_decay)`
   - Loss: `nn.MSELoss()`

3. **Training loop** (up to `max_epochs`):
   - For each epoch: train loop → compute train loss → compute val loss (no grad)
   - Track `self.train_losses_: list[float]` and `self.val_losses_: list[float]`
   - **Early stopping**: if val loss does not improve for `early_stopping_patience` epochs, stop and restore the best weights
   - Use `tqdm` progress bar showing current train/val loss

4. **Store training metadata**:
   - `self.best_epoch_: int`
   - `self.final_train_loss_: float`
   - `self.final_val_loss_: float`
   - `self.n_assets_: int`
   - `self.tickers_: list[str]`

5. Set `self._is_fitted = True`

6. Return training history dict:
   ```python
   {
       "train_losses": list[float],
       "val_losses": list[float],
       "best_epoch": int,
       "stopped_early": bool
   }
   ```

---

### Method: `get_residuals(self, returns: pd.DataFrame) -> pd.DataFrame`

- Set network to eval mode (`self.network_.eval()`)
- Pass returns through the full autoencoder (no grad)
- Residuals: `epsilon_{i,t} = R_{i,t} - R_hat_{i,t}` (actual minus reconstruction)
- Return DataFrame with same shape/index as input

---

### Method: `get_factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame`

- Returns the **reconstruction** `R_hat` — the systematic (market factor) component
- Same shape as input

---

### Method: `get_bottleneck_representation(self, returns: pd.DataFrame) -> pd.DataFrame`

- Pass returns through encoder only
- Return shape `(T, bottleneck)` — the latent factor representation
- Column names: `["factor_0", "factor_1", ..., "factor_{k-1}"]`

---

### Method: `save(self, path: str) -> None` and `load(cls, path: str, config: dict) -> AutoencoderModel`

- Save the full model state (network weights + metadata) using `torch.save`
- `load` is a `@classmethod` that reconstructs the model from disk

---

## 4.3 Activation Function Comparison

When performing the activation function grid search, each activation is trained under identical conditions (same seed, same data). The `fit()` method must accept and use `torch.manual_seed(config['random_seed'])` before initializing weights, so results are reproducible and comparable.
