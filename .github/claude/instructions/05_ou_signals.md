# Instruction 05 — Ornstein-Uhlenbeck Process & Signal Generation

All signal logic lives in `stat_arb/signals/ou_process.py`.

This module is the key shared component: it takes residuals (from either PCA or the Autoencoder) and produces trading signals. It must be completely model-agnostic.

---

## 5.1 `OUProcess` — Per-Stock Parameter Estimation

### Class: `OUProcess`

```
OUProcess(config: dict)
```

Stores:
- `ou_lookback_days: int` from `config['signals']['ou_lookback_days']`
- `min_kappa: float` from `config['signals']['min_kappa']`

---

### Method: `estimate_parameters(self, residuals: np.ndarray) -> dict`

Estimate OU parameters from a 1D array of residual returns (a single stock's residual series over the lookback window).

**Implementation — exactly follows Avellaneda & Lee Appendix A**:

Step 1 — Cumulative residual process:
- `X_k = cumsum(residuals)` for `k = 1, ..., T`
- Note: X_T = 0 by construction of OLS regression (the regression forces residuals to sum to zero over the window). This is a known artifact.

Step 2 — AR(1) regression:
- Regress `X_{n+1}` on `X_n` using OLS (via `numpy.linalg.lstsq` or `statsmodels.OLS`):
  - `X_{n+1} = a + b * X_n + noise`
- Extract `a`, `b`, and `var(noise)` (residual variance of the AR regression)

Step 3 — Recover OU parameters (using the discrete-time mapping in Appendix A):
```
kappa = -log(b) * 252          # annualized speed of mean reversion
m     = a / (1 - b)            # long-run mean
sigma = sqrt(var_noise * 2 * kappa / (1 - b**2))   # diffusion coefficient
sigma_eq = sqrt(var_noise / (1 - b**2))             # equilibrium std
```

Step 4 — Validity check:
- If `b >= 1` (unit root) or `b <= 0` (oscillating), mark as **invalid**
- If `kappa < min_kappa` (too slow to mean-revert), mark as **invalid**

Return dict:
```python
{
    "a": float,
    "b": float,
    "kappa": float,          # annualized
    "m": float,              # long-run mean
    "sigma": float,          # diffusion coeff
    "sigma_eq": float,       # equilibrium std (denominator of s-score)
    "var_noise": float,
    "is_valid": bool         # False if kappa < min_kappa or b >= 1
}
```

---

## 5.2 `ZScoreGenerator` — Rolling Z-Score Computation

### Class: `ZScoreGenerator`

```
ZScoreGenerator(config: dict)
```

---

### Method: `compute_zscores(self, residuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`

**Input**: Residual DataFrame `(T, N)` — one column per stock, index = dates

**Output**: Tuple of:
1. `zscores: pd.DataFrame` — shape `(T, N)`, Z-score for each stock at each time step
2. `ou_params: pd.DataFrame` — shape `(T, N)`, the OU `kappa` value (useful for diagnostics; a row per day, column per stock)

**Algorithm** — rolling window, computed daily:

For each time step `t` (from `ou_lookback_days` onward):
1. Extract the lookback window: `residuals.iloc[t - ou_lookback_days : t]` — shape `(lookback, N)`
2. For each stock `i`:
   a. Call `OUProcess.estimate_parameters(window[:, i])`
   b. If `is_valid = False`, set `z_{i,t} = NaN` (do not trade this stock on this day)
   c. If valid, compute the **centred s-score** (Avellaneda & Lee equation A.2):

      The s-score is:
      ```
      s_{i,t} = -m_i / sigma_eq_i
      ```
      This is because at the end of the estimation window `X_T = 0` (by OLS construction), so:
      ```
      s = (X(t) - m) / sigma_eq = (0 - m) / sigma_eq = -m / sigma_eq
      ```

      The **centred** version (which consistently outperformed in Avellaneda & Lee) further subtracts the cross-sectional mean of `m / sigma_eq`:
      ```
      s_centred_{i,t} = -m_i / sigma_eq_i + mean_j(m_j / sigma_eq_j)
      ```

3. Fill the zscores DataFrame at row `t`

For `t < ou_lookback_days`, fill all entries with `NaN`.

**Performance note**: This double loop (over T and N) is the computational bottleneck. Optimize with `numpy` vectorization where possible. For the grid search, consider memoizing residuals and only recomputing z-scores when OU params change.

---

## 5.3 `SignalGenerator` — Trading Signal from Z-Scores

### Class: `SignalGenerator`

```
SignalGenerator(
    config: dict,
    zscore_entry: float | None = None,
    zscore_exit: float | None = None
)
```

Default thresholds from config if None.

---

### Method: `generate_signals(self, zscores: pd.DataFrame) -> pd.DataFrame`

**Input**: Z-score DataFrame `(T, N)`

**Output**: Signal DataFrame `(T, N)` with values in `{-1, 0, 1}`
- `+1` = long (stock is undervalued: z < -entry_threshold)
- `-1` = short (stock is overvalued: z > +entry_threshold)
- `0`  = no position

**Signal logic (stateful — requires iterating row by row)**:

Maintain a position state for each stock. Initialize all positions to 0.

For each time step `t`:
- For each stock `i`:
  - If current position = 0 (no trade open):
    - If `z_{i,t} < -zscore_entry` → open long → set position = +1
    - If `z_{i,t} > +zscore_entry` → open short → set position = -1
  - If current position = +1 (long):
    - If `z_{i,t} > -zscore_exit` → close long → set position = 0  *(Z has reverted past exit)*
    - If `z_{i,t} > +zscore_entry` → flip to short → set position = -1
  - If current position = -1 (short):
    - If `z_{i,t} < +zscore_exit` → close short → set position = 0
    - If `z_{i,t} < -zscore_entry` → flip to long → set position = +1
  - If `z_{i,t}` is `NaN` → close position (invalid OU fit) → set position = 0

Return the signal DataFrame (each cell is the position **at end of day t**, acted on the **next** day's open — this avoids look-ahead bias in backtesting).

---

## 5.4 Important Detail: Preventing Look-Ahead Bias

The signal computed on day `t` is based on residuals up to and including day `t`. This signal must be applied to **day t+1's returns** in the backtesting engine. The `SignalGenerator` returns signals indexed by the decision date; the `BacktestEngine` must shift them by 1 day when computing PnL.
