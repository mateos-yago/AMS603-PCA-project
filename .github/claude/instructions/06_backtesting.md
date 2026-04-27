# Instruction 06 — Backtesting Engine

All backtesting logic lives in `stat_arb/backtesting/`. The engine is model-agnostic — it takes a signal DataFrame and raw returns, and simulates portfolio performance.

---

## 6.1 `transaction_costs.py` — `TransactionCostModel`

### Class: `TransactionCostModel`

```
TransactionCostModel(config: dict)
```

Stores:
- `cost_bps: float` from `config['transaction_costs']['cost_bps']`
- `bid_ask_bps: float` from `config['transaction_costs']['bid_ask_spread_bps']`
- `total_one_way_bps: float = cost_bps + bid_ask_bps`

---

### Method: `compute_costs(self, position_changes: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.Series`

**Input**:
- `position_changes`: DataFrame `(T, N)` — absolute change in position weight per stock per day
- `prices`: optional, not needed if we work in weight-space

**Output**: `pd.Series` of shape `(T,)` — total transaction cost as a fraction of portfolio value per day

**Formula**:
```
cost_t = sum_i |delta_weight_{i,t}| * total_one_way_bps / 10000
```

The cost is deducted from portfolio returns on each day a trade occurs.

---

## 6.2 `portfolio.py` — `DollarNeutralPortfolio`

### Class: `DollarNeutralPortfolio`

```
DollarNeutralPortfolio(config: dict)
```

---

### Method: `compute_weights(self, signals: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Convert raw signals `{-1, 0, +1}` into dollar-neutral portfolio weights.

**Algorithm**:
1. For each time step `t`:
   - Identify long stocks: `signals.iloc[t] == +1`
   - Identify short stocks: `signals.iloc[t] == -1`
   - Count: `n_long`, `n_short`
   - If either is zero, set all weights to 0 (cannot form a dollar-neutral portfolio)
   - Long weight per stock: `leverage / (2 * n_long)` (positive)
   - Short weight per stock: `-leverage / (2 * n_short)` (negative)
   - This ensures: `sum(long weights) = leverage/2` and `sum(short weights) = -leverage/2`
   - Verify: `sum(all weights) ≈ 0` (dollar-neutral)

2. Cap individual weights at `config['portfolio']['max_position_weight']`:
   - If a stock's weight exceeds the cap, reduce it and redistribute excess pro-rata to other long/short positions

**Output**: Weight DataFrame `(T, N)` with values summing to ~0 at each row.

---

## 6.3 `engine.py` — `BacktestEngine`

### Class: `BacktestEngine`

```
BacktestEngine(config: dict)
```

This is the core simulation. It is completely model-agnostic.

---

### Method: `run(self, signals: pd.DataFrame, raw_returns: pd.DataFrame, apply_costs: bool | None = None) -> dict`

**Inputs**:
- `signals`: DataFrame `(T, N)` — output of `SignalGenerator.generate_signals()`, indexed by decision date
- `raw_returns`: DataFrame `(T, N)` — actual (non-standardized) daily log returns for the test period
- `apply_costs`: bool — if None, read from `config['backtesting']['apply_transaction_costs']`

**Critical**: Signals from day `t` are applied to returns on day `t+1`. Implement this by shifting signals forward by 1 day:
```python
weights = portfolio.compute_weights(signals)
weights_shifted = weights.shift(1).fillna(0)   # day t+1 position is determined by day t signal
```

**PnL calculation** (daily):
```
portfolio_return_t = sum_i weight_{i, t-1} * raw_return_{i,t}   - transaction_cost_t
```

where `transaction_cost_t = TransactionCostModel.compute_costs(|weights_t - weights_{t-1}|)`.

**Output dict**:
```python
{
    "daily_returns": pd.Series,            # portfolio return each day (T,)
    "cumulative_returns": pd.Series,       # cumulative product (T,)
    "weights": pd.DataFrame,               # portfolio weights (T, N)
    "transaction_costs": pd.Series,        # cost per day (T,)
    "n_long": pd.Series,                   # number of long positions per day
    "n_short": pd.Series,                  # number of short positions per day
    "gross_exposure": pd.Series,           # sum of |weights| per day
}
```

---

### Method: `run_with_and_without_costs(self, signals: pd.DataFrame, raw_returns: pd.DataFrame) -> dict`

Convenience method that calls `run()` twice (with and without costs) and returns both result dicts:
```python
{
    "with_costs": dict,
    "without_costs": dict
}
```

This is required because the project must show both scenarios side by side.

---

## 6.4 Complete Backtest Pipeline

A convenience function (not a class) in `engine.py`:

```python
def run_full_backtest(
    factor_model: BaseFactorModel,
    raw_returns_test: pd.DataFrame,
    std_returns_test: pd.DataFrame,
    config: dict,
    zscore_entry: float | None = None,
    zscore_exit: float | None = None,
) -> dict:
    """
    End-to-end backtest:
    1. Get residuals from the fitted factor model
    2. Compute Z-scores via OU process
    3. Generate signals
    4. Run backtest (with and without costs)
    5. Return combined results dict
    """
```

This function is what the notebooks call. It encapsulates the entire residual → signal → backtest pipeline.
