# Instruction 06b — Performance Metrics

All metric computation lives in `stat_arb/metrics/performance.py`.

---

## Class: `PerformanceAnalyzer`

```
PerformanceAnalyzer(config: dict)
```

Stores `risk_free_rate_annual` from config.

---

## Methods

All methods accept a `pd.Series` of **daily** portfolio returns (not log returns — use simple returns for PnL).

### `annualized_return(self, daily_returns: pd.Series) -> float`
```
(1 + mean_daily_return)^252 - 1
```

### `annualized_volatility(self, daily_returns: pd.Series) -> float`
```
daily_std * sqrt(252)
```

### `sharpe_ratio(self, daily_returns: pd.Series) -> float`
```
(annualized_return - risk_free_rate) / annualized_volatility
```
Use daily risk-free: `rf_daily = (1 + rf_annual)^(1/252) - 1`
Then: `Sharpe = (mean(daily_returns - rf_daily) / std(daily_returns)) * sqrt(252)`

### `maximum_drawdown(self, daily_returns: pd.Series) -> float`
- Compute cumulative wealth: `(1 + daily_returns).cumprod()`
- Rolling max of wealth
- Drawdown at each point: `(wealth - rolling_max) / rolling_max`
- Return the minimum (most negative) value

### `max_drawdown_duration(self, daily_returns: pd.Series) -> int`
- Return the number of days in the longest drawdown period (from peak to recovery)

### `hit_ratio(self, daily_returns: pd.Series) -> float`
- Fraction of trading days with positive returns: `sum(daily_returns > 0) / len(daily_returns)`

### `turnover_rate(self, weights: pd.DataFrame) -> float`
- Average daily one-way turnover: `mean(sum_i |weight_{i,t} - weight_{i,t-1}|)` over all t
- Annualize: multiply by 252

### `calmar_ratio(self, daily_returns: pd.Series) -> float`
```
annualized_return / abs(maximum_drawdown)
```

### `compute_all(self, daily_returns: pd.Series, weights: pd.DataFrame) -> dict`

Returns a single dict with all metrics:
```python
{
    "annualized_return": float,
    "annualized_volatility": float,
    "sharpe_ratio": float,
    "maximum_drawdown": float,
    "max_drawdown_duration_days": int,
    "hit_ratio": float,
    "daily_turnover": float,
    "annualized_turnover": float,
    "calmar_ratio": float,
    "total_return": float,
    "n_trading_days": int,
}
```

### `compare(self, results_a: dict, results_b: dict, label_a: str, label_b: str) -> pd.DataFrame`

Creates a formatted comparison table (DataFrame) of all metrics side by side.
Useful for the final notebook comparison.
