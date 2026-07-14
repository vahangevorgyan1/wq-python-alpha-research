# WQ Alpha Research

This project keeps three high-performing WorldQuant BRAIN Python alpha ideas
with their implementation notes, simulation settings, and research rationale.

The alphas are stored as standalone Python files under `alphas/`. Each file
contains the full BRAIN Python alpha code plus a `SETTINGS` dictionary that
records the simulation configuration used for the reported result.

## Alphas

### 1. IND Positive Intraday Hit-Rate Reversal

File: `alphas/ind_positive_intraday_hit_rate_reversal.py`

Core idea:
measure how often a stock has a positive open-to-close return over the last
250 trading days, then subtract today's cross-sectional intraday return rank.

```text
hit_rate = ts_sum(open_to_close_return > 0, 250) / 250
signal = (hit_rate - 0.5) - rank(today_open_to_close_return)
```

Settings:

- Region/universe: `IND TOP500`
- Neutralization: `STATISTICAL`
- Decay/delay/truncation: `25 / 1 / 0.08`
- Lookback: `251`

Reported result:

- Sharpe `4.32`
- Fitness `3.43`
- Turnover `33.96%`
- Returns `21.41%`
- Drawdown `4.01%`
- Margin `12.61%`

Why it can work:
this is a reversal signal conditioned on persistent intraday directionality.
The hit-rate term captures whether a stock frequently closes above its open,
while subtracting today's rank fades names that already had strong same-day
movement. In `IND TOP500`, this can exploit short-horizon overreaction while
statistical neutralization removes broad market/style effects.

Possible improvements:

- Test `close/open - 1` against `close/vwap - 1` and `vwap/open - 1`.
- Try 120-day or 180-day hit-rate windows for faster adaptation.
- Add liquidity or volatility buckets if turnover/capacity becomes an issue.
- Vectorize the positive-return calculation if runtime becomes important.

### 2. IND Midpoint-Close Rank Residual

File: `alphas/ind_midpoint_close_rank_residual.py`

Core idea:
rank the current value of `(high + low) / 2 - close` inside its own 252-day
history, then neutralize the signal by subindustry in Python.

```text
raw = ts_rank(((high + low) / 2) - close, 252)
signal = group_neutralize(raw, subindustry)
```

Settings:

- Region/universe: `IND TOP500`
- Neutralization: `STATISTICAL`
- Decay/delay/truncation: `20 / 1 / 0.08`
- Lookback: `252`

Reported result:

- Sharpe `3.93`
- Fitness `2.81`
- Turnover `34.57%`
- Returns `17.67%`
- Drawdown `3.67%`
- Margin `10.22%`

Why it can work:
the close relative to the day's midpoint shows whether a stock finished near
the top or bottom of its trading range. A time-series rank makes this
stock-specific and robust to each name's normal volatility. Subindustry
neutralization then removes peer-group structure, leaving a cleaner residual
price-location signal.

Possible improvements:

- Replace the per-instrument `ts_rank_last` loop with a vectorized rank helper.
- Try 126-day or 180-day windows to reduce lookback and runtime.
- Add a volatility regime gate for noisy high-volatility periods.
- Compare subindustry neutralization against industry or sector neutralization.

### 3. MEA VWAP Dislocation Residual

File: `alphas/mea_vwap_dislocation_residual.py`

Core idea:
backfill VWAP, measure the normalized VWAP-close spread, compare recent values
with a trailing baseline, apply a signed power transform, then rank and
industry-neutralize.

```text
ratio = (backfill(vwap, 5) - close) / backfill(vwap, 5)
shock = signed_power(ratio_t - mean(ratio), 0.65)
signal = group_neutralize(rank(weighted_recent_shock), industry)
```

Settings:

- Region/universe: `MEA TOP400`
- Neutralization: `COUNTRY`
- Decay/delay/truncation: `20 / 1 / 0.08`
- Lookback: `252`

Reported result:

- Sharpe `3.24`
- Fitness `2.60`
- Turnover `27.63%`
- Returns `17.81%`
- Drawdown `4.95%`
- Margin `12.89%`

Why it can work:
VWAP-close dislocation is a proxy for intraday execution pressure and flow
imbalance. When the close is unusually far from VWAP relative to its recent
baseline, the stock may be temporarily mispriced. Ranking makes the signal
cross-sectionally robust, while industry and country neutralization reduce
regional and sector concentration.

Possible improvements:

- Compare VWAP-close spread with open-close spread.
- Test 40, 60, and 90-day baselines instead of the current 75-day averages.
- Add liquidity buckets if the signal becomes concentrated.
- Try sector/subindustry code-neutralization while keeping country
  neutralization in settings.
- Rewrite `ts_backfill` with a more vectorized helper if runtime matters.
