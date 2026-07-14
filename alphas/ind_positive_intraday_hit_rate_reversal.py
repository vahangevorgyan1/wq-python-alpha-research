"""IND positive intraday hit-rate reversal.

Reported setup:
IND TOP500, decay 25, delay 1, truncation 0.08,
neutralization STATISTICAL, lookback 251.

Reported result:
Sharpe 4.32, Fitness 3.43, Turnover 33.96%,
Returns 21.41%, Drawdown 4.01%, Margin 12.61%.
"""

from brain.alphas import alpha
import numpy as np
import numpy.typing as npt


SETTINGS = {
    "region": "IND",
    "universe": "TOP500",
    "decay": 25,
    "delay": 1,
    "truncation": 0.08,
    "neutralization": "STATISTICAL",
    "pasteurization": "ON",
    "language": "PYTHON",
    "lookback": 251,
}


def pasteurize(x, universe):
    return np.where(universe.astype(bool), x, np.nan).astype(np.float32)


def cs_rank(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    good = np.isfinite(x)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    n = int(np.sum(good))
    if n > 1:
        order = np.argsort(x[good], kind="mergesort")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[good] = ranks
    return out.astype(np.float32)


def clean_div(num, den):
    out = np.where(
        np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12),
        num / den,
        np.nan,
    )
    return out.astype(np.float32)


def lagged_returns(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    ret = clean_div(x[1:], x[:-1]) - 1.0
    ret = np.where(np.isfinite(ret), ret, np.nan)
    return ret.astype(np.float32)


def intraday_returns(open_, close):
    ret = clean_div(close, open_) - 1.0
    ret = np.where(np.isfinite(ret), ret, np.nan)
    return ret.astype(np.float32)


def ts_sum_positive_returns(x: npt.NDArray[np.float32], d: int) -> npt.NDArray[np.float32]:
    window = x[-int(d):]
    positive = np.where(np.isfinite(window), np.where(window > 0, 1.0, 0.0), np.nan)
    out = np.nansum(positive, axis=0)
    out = np.where(np.any(np.isfinite(positive), axis=0), out, np.nan)
    return out.astype(np.float32)


@alpha(data=["open", "close"], store=[])
def posfreq_open_close_return_minus_rank(data, store) -> npt.NDArray[np.float32]:
    r = intraday_returns(data.open.astype(np.float32), data.close.astype(np.float32))
    hit_rate = ts_sum_positive_returns(r, 250) / 250.0
    signal = (hit_rate - 0.5) - cs_rank(r[-1])
    return pasteurize(signal.astype(np.float32), data.universe[-1])
