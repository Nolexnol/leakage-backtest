"""Performance statistics that respect autocorrelation and multiple testing."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio of a daily return series (zero risk-free rate)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """t-statistic of the mean of ``x`` with Newey-West (HAC) standard errors.

    Daily IC series and overlapping-horizon returns are autocorrelated, so the
    naive t-stat overstates significance.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    gamma0 = e @ e / n
    lrv = gamma0
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        lrv += 2 * w * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / math.sqrt(lrv / n))


def stationary_bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_boot: int = 2000,
    mean_block: int = 10,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for the annualized Sharpe using the Politis-Romano stationary bootstrap.

    Resampling random-length blocks keeps short-range autocorrelation intact.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(n)
        new_block = rng.random(n) < p
        starts = rng.integers(n, size=n)
        for i in range(1, n):
            idx[i] = starts[i] if new_block[i] else (idx[i - 1] + 1) % n
        out[b] = sharpe(r[idx])
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``), Bailey & Lopez de Prado (2012).

    Corrects for sample length, skewness and fat tails. ``sr_benchmark`` is per
    period (daily), not annualized.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    sr = r.mean() / r.std(ddof=1)
    skew = sps.skew(r)
    kurt = sps.kurtosis(r, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return float(sps.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Expected maximum per-period Sharpe among N skill-less trials.

    From Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio". Uses the
    cross-trial variance of the (per-period) Sharpe estimates.
    """
    s = np.asarray(trial_sharpes, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n < 2:
        return 0.0
    sd = s.std(ddof=1)
    z1 = sps.norm.ppf(1 - 1.0 / n)
    z2 = sps.norm.ppf(1 - 1.0 / (n * math.e))
    return float(sd * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(best_returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Probability that the best of N trials has a true Sharpe above zero,
    after accounting for having picked it out of N attempts."""
    return probabilistic_sharpe_ratio(best_returns, sr_benchmark=expected_max_sharpe(trial_sharpes))
