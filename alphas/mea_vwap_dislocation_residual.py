"""MEA VWAP dislocation residual.

Reported setup:
MEA TOP400, decay 20, delay 1, truncation 0.08,
neutralization COUNTRY, lookback 252.

Reported result:
Sharpe 3.24, Fitness 2.60, Turnover 27.63%,
Returns 17.81%, Drawdown 4.95%, Margin 12.89%.
"""

from brain.alphas import alpha
import numpy as np
import numpy.typing as npt


SETTINGS = {
    "region": "MEA",
    "universe": "TOP400",
    "decay": 20,
    "delay": 1,
    "truncation": 0.08,
    "neutralization": "COUNTRY",
    "pasteurization": "ON",
    "language": "PYTHON",
    "lookback": 252,
}


def ts_backfill(x: npt.NDArray[np.float32], d: int) -> npt.NDArray[np.float32]:
    out = x.copy()
    for lag in range(1, d):
        shifted = np.empty_like(x)
        shifted[:lag] = np.nan
        shifted[lag:] = x[:-lag]
        out = np.where(np.isfinite(out), out, shifted)
    return out.astype(np.float32)


def signed_power(x: npt.NDArray[np.float32], p: float) -> npt.NDArray[np.float32]:
    y = np.sign(x) * np.power(np.abs(x), p)
    return np.where(np.isfinite(y), y, np.nan).astype(np.float32)


def cs_rank(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    good = np.isfinite(x)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    n = int(np.sum(good))
    if n > 1:
        order = np.argsort(x[good])
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[good] = ranks
    return out


def group_neutralize(x: npt.NDArray[np.float32], group: npt.NDArray) -> npt.NDArray[np.float32]:
    g = group[-1] if getattr(group, "ndim", 1) > 1 else group
    try:
        g = g.astype(np.float32)
        group_good = np.isfinite(g)
    except Exception:
        group_good = np.array([v is not None for v in g])
    good = np.isfinite(x) & group_good
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if np.sum(good) < 2:
        return out
    _, inv = np.unique(g[good], return_inverse=True)
    sums = np.bincount(inv, weights=x[good])
    counts = np.bincount(inv)
    means = sums / counts
    out[good] = (x[good] - means[inv]).astype(np.float32)
    return out


def pasteurize(a: npt.NDArray[np.float32], universe: npt.NDArray) -> npt.NDArray[np.float32]:
    a = a.copy()
    a[~universe.astype(bool)] = np.nan
    return a.astype(np.float32)


def base_signal(vwap, close) -> npt.NDArray[np.float32]:
    vwap_bf = ts_backfill(vwap, 5)
    ratio = (vwap_bf - close) / vwap_bf
    ratio = np.where(np.isfinite(ratio), ratio, np.nan).astype(np.float32)
    mean_t1 = np.nanmean(ratio[-76:-1], axis=0)
    mean_t0 = np.nanmean(ratio[-75:], axis=0)
    x_t1 = ratio[-2] - mean_t1
    x_t0 = ratio[-1] - mean_t0
    sp_t1 = signed_power(x_t1.astype(np.float32), 0.65)
    sp_t0 = signed_power(x_t0.astype(np.float32), 0.65)
    return ((sp_t1 + 2.0 * sp_t0) / 3.0).astype(np.float32)


@alpha(data=["vwap", "close", "industry"], store=[])
def vwap_dislocation_mea(data, store) -> npt.NDArray[np.float32]:
    decayed = base_signal(data.vwap, data.close)
    signal = cs_rank(decayed.astype(np.float32))
    signal = group_neutralize(signal.astype(np.float32), data.industry)
    return pasteurize(signal.astype(np.float32), data.universe[-1])
