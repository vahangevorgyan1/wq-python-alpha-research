"""IND midpoint-close rank residual.

Reported setup:
IND TOP500, decay 20, delay 1, truncation 0.08,
neutralization STATISTICAL, lookback 252.

Reported result:
Sharpe 3.93, Fitness 2.81, Turnover 34.57%,
Returns 17.67%, Drawdown 3.67%, Margin 10.22%.
"""

from brain.alphas import alpha
import numpy as np
import numpy.typing as npt


SETTINGS = {
    "region": "IND",
    "universe": "TOP500",
    "decay": 20,
    "delay": 1,
    "truncation": 0.08,
    "neutralization": "STATISTICAL",
    "pasteurization": "ON",
    "language": "PYTHON",
    "lookback": 252,
}


def pasteurize(x, universe):
    return np.where(universe.astype(bool), x, np.nan).astype(np.float32)


def ts_rank_last(x: npt.NDArray[np.float32], d: int) -> npt.NDArray[np.float32]:
    window = x[-int(d):]
    last = window[-1]
    out = np.full(last.shape, np.nan, dtype=np.float32)
    for i in range(last.shape[0]):
        vals = window[:, i]
        good = np.isfinite(vals)
        if not np.isfinite(last[i]) or np.sum(good) < 2:
            continue
        v = vals[good]
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(len(v), dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(v), dtype=np.float32)
        last_pos = np.where(good)[0][-1]
        compressed_pos = np.sum(good[: last_pos + 1]) - 1
        out[i] = ranks[compressed_pos]
    return out.astype(np.float32)


def group_neutralize(x: npt.NDArray[np.float32], group) -> npt.NDArray[np.float32]:
    g = group[-1] if getattr(group, "ndim", 1) > 1 else group
    g = np.asarray(g)
    group_good = np.array([v is not None and str(v) != "nan" for v in g])
    good = np.isfinite(x) & group_good
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if np.sum(good) < 2:
        return out
    _, inv = np.unique(g[good], return_inverse=True)
    sums = np.bincount(inv, weights=x[good])
    counts = np.bincount(inv)
    means = sums / counts
    out[good] = x[good] - means[inv]
    return out.astype(np.float32)


@alpha(
    data=[
        "high",
        "low",
        "close",
        "subindustry",
    ],
    store=[],
)
def midpoint_close_tsrank_subindustry_neut(data, store) -> npt.NDArray[np.float32]:
    x = ((data.high + data.low) / 2.0) - data.close
    signal = ts_rank_last(x.astype(np.float32), 252)
    signal = group_neutralize(signal, data.subindustry)
    return pasteurize(signal, data.universe[-1])
