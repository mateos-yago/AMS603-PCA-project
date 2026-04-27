# Instruction 03 — PCA Factor Model

All PCA logic lives in `stat_arb/factors/`. Start by implementing the abstract base class, then the PCA model.

---

## 3.1 Abstract Base Class — `base_factor_model.py`

### Class: `BaseFactorModel` (ABC)

```
BaseFactorModel(config: dict)
```

This is the contract that both `PCAModel` and `AutoencoderModel` must fulfill. Define the following abstract methods:

```python
@abstractmethod
def fit(self, returns: pd.DataFrame) -> None:
    """Fit the model on training data."""

@abstractmethod
def get_residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Given a returns DataFrame, return the idiosyncratic residuals.
    Shape: same as input (T, N).
    """

@abstractmethod
def get_factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Return the reconstructed/systematic component of returns.
    Shape: same as input (T, N).
    """
```

Also implement a concrete method:

```python
def validate_not_fitted(self) -> None:
    """Raise RuntimeError if the model has not been fitted yet."""
```

Use a `self._is_fitted: bool = False` flag that subclasses set to `True` at the end of `fit()`.

---

## 3.2 `pca_model.py` — `PCAModel`

### Class: `PCAModel(BaseFactorModel)`

```
PCAModel(config: dict, n_factors: int | None = None)
```

If `n_factors` is None, read from `config['pca']['default_n_factors']`.

---

### Method: `fit(self, returns: pd.DataFrame) -> None`

**This is the core Avellaneda & Lee PCA procedure.**

Step 1 — Compute the empirical correlation matrix:
- Input shape: `(T, N)` where T = training days, N = assets
- Per-stock, compute time-series mean `R_bar_i` and std `sigma_i` over the full training window
- Compute standardized returns `Y_ik = (R_ik - R_bar_i) / sigma_i`
- Compute correlation matrix `Sigma = (1/(T-1)) * Y.T @ Y`  — shape `(N, N)`

Step 2 — Eigendecomposition:
- Use `numpy.linalg.eigh` (symmetric matrix, returns sorted eigenvalues in ascending order — reverse them)
- Keep the top `n_factors` eigenvectors. Store:
  - `self.eigenvalues_: np.ndarray` — shape `(n_factors,)`
  - `self.eigenvectors_: np.ndarray` — shape `(N, n_factors)`, each column is an eigenvector
  - `self.explained_variance_ratio_: np.ndarray` — fraction of total variance explained by each factor
  - `self.cumulative_variance_explained_: float` — total fraction explained by top k factors

Step 3 — Compute eigenportfolio weights:
- Per Avellaneda & Lee equation (9): `Q^(j)_i = v^(j)_i / sigma_i`
- Store `self.eigenportfolio_weights_: np.ndarray` — shape `(N, n_factors)`

Step 4 — Compute factor betas (OLS regression of each stock on the k factor returns):
- Factor returns `F_{j,t} = sum_i Q^(j)_i * R_{i,t}` — shape `(T, n_factors)`
- For each stock i, regress `R_{i,t}` on `[1, F_{1,t}, ..., F_{k,t}]` using `numpy.linalg.lstsq`
- Store `self.betas_: np.ndarray` — shape `(N, n_factors+1)` — first column is intercept
- Store `self.factor_returns_train_: pd.DataFrame` — shape `(T, n_factors)`

Set `self._is_fitted = True`.

---

### Method: `get_factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame`

- Compute factor returns from held-out data using the stored eigenportfolio weights
- `F_{j,t} = sum_i Q^(j)_i * R_{i,t}`
- Return shape `(T, n_factors)`

---

### Method: `get_residuals(self, returns: pd.DataFrame) -> pd.DataFrame`

- For each stock i, compute the systematic component: `R_hat_{i,t} = beta_{i,0} + sum_j beta_{i,j} * F_{j,t}`
- Residual: `epsilon_{i,t} = R_{i,t} - R_hat_{i,t}`
- Return DataFrame with same shape and index as input

---

### Additional stored attributes (set during `fit`):

```python
self.tickers_: list[str]             # list of asset names in order
self.n_assets_: int
self.n_factors_: int
self.per_stock_mean_: np.ndarray     # R_bar_i, shape (N,)
self.per_stock_std_: np.ndarray      # sigma_i, shape (N,)
self.correlation_matrix_: np.ndarray # shape (N, N)
```

---

### Method: `get_variance_explained_summary(self) -> dict`

Returns:
```python
{
    "n_factors": int,
    "eigenvalues": list[float],
    "explained_variance_ratio": list[float],
    "cumulative_variance_explained": float
}
```

---

## 3.3 Grid Search for PCA

Grid search is handled externally in `stat_arb/experiments/pca_grid_search.py`. The `PCAModel` class itself does not contain grid search logic — it must be instantiable with arbitrary `n_factors`.

See `instructions/07_experiments.md` for grid search details.
