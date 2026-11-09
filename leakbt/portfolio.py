"""Long-short decile backtest and daily rank IC.

Uses the C++ kernels in ``leakbt._fastbt`` when they are built
(``pip install -e .``) and falls back to equivalent NumPy code otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import stats

try:  # pragma: no cover - depends on whether the extension was compiled
    from . import _fastbt  # type: ignore

    HAVE_CPP = True
except ImportError:  # pragma: no cover
    _fastbt = None
    HAVE_CPP = False


def long_short_numpy(scores: np.ndarray, returns: np.ndarray, quantile: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    T, N = scores.shape
    pnl = np.full(T, np.nan)
    turnover = np.zeros(T)
    w_prev = np.zeros(N)
    for t in range(T):
        w = np.zeros(N)
        valid = np.flatnonzero(np.isfinite(scores[t]))
        m = valid.size
        k = int(np.floor(quantile * m))
        if k >= 1:
            order = valid[np.argsort(scores[t, valid], kind="stable")]
            w[order[:k]] = -0.5 / k
            w[order[m - k:]] = 0.5 / k
            r = np.where(np.isfinite(returns[t]), returns[t], 0.0)
            pnl[t] = float(w @ r)
        turnover[t] = np.abs(w - w_prev).sum()
        w_prev = w
    return pnl, turnover


def rank_ic_numpy(scores: np.ndarray, returns: np.ndarray) -> np.ndarray:
    T = scores.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        ok = np.isfinite(scores[t]) & np.isfinite(returns[t])
        if ok.sum() < 10:
            continue
        a, b = rankdata(scores[t, ok]), rankdata(returns[t, ok])
        if a.std() == 0 or b.std() == 0:
            continue
        out[t] = np.corrcoef(a, b)[0, 1]
    return out


def long_short(scores: np.ndarray, returns: np.ndarray, quantile: float = 0.1, use_cpp: bool = True):
    s = np.ascontiguousarray(scores, dtype=float)
    r = np.ascontiguousarray(returns, dtype=float)
    if use_cpp and HAVE_CPP:
        pnl, turnover = _fastbt.long_short(s, r, quantile)
        return np.asarray(pnl), np.asarray(turnover)
    return long_short_numpy(s, r, quantile)


def rank_ic(scores: np.ndarray, returns: np.ndarray, use_cpp: bool = True) -> np.ndarray:
    s = np.ascontiguousarray(scores, dtype=float)
    r = np.ascontiguousarray(returns, dtype=float)
    if use_cpp and HAVE_CPP:
        return np.asarray(_fastbt.rank_ic(s, r))
    return rank_ic_numpy(s, r)


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    gross: np.ndarray
    net: np.ndarray
    turnover: np.ndarray
    ic: np.ndarray

    def summary(self) -> dict:
        lo, hi = stats.stationary_bootstrap_sharpe_ci(self.net)
        return {
            "days": int(np.isfinite(self.gross).sum()),
            "mean_ic": float(np.nanmean(self.ic)),
            "ic_tstat_nw": stats.newey_west_tstat(self.ic),
            "sharpe_gross": stats.sharpe(self.gross),
            "sharpe_net": stats.sharpe(self.net),
            "sharpe_net_ci": (lo, hi),
            "ann_return_net": float(np.nanmean(self.net) * stats.TRADING_DAYS),
            "avg_daily_turnover": float(np.nanmean(self.turnover)),
            "psr_net": stats.probabilistic_sharpe_ratio(self.net),
        }


def run_backtest(
    scores: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    quantile: float = 0.1,
    cost_bps: float = 5.0,
    use_cpp: bool = True,
) -> BacktestResult:
    """Daily-rebalanced long-short backtest.

    ``scores`` and ``fwd_returns`` are aligned wide frames (date x ticker) where the
    row for date t holds the signal formed at the close of t and the 1-day return
    earned by trading on it. ``cost_bps`` is charged per unit of turnover (one-way).
    """
    fwd_returns = fwd_returns.reindex(index=scores.index, columns=scores.columns)
    s, r = scores.to_numpy(), fwd_returns.to_numpy()
    gross, turnover = long_short(s, r, quantile, use_cpp)
    net = gross - turnover * cost_bps / 1e4
    ic = rank_ic(s, r, use_cpp)
    return BacktestResult(scores.index, gross, net, turnover, ic)
