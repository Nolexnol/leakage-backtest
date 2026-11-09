import math

import numpy as np
import pytest
from scipy.stats import spearmanr

from leakbt import portfolio, stats


def _random(T=120, N=60, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((T, N))
    r = rng.standard_normal((T, N)) * 0.01
    s[rng.random((T, N)) < 0.05] = np.nan
    r[rng.random((T, N)) < 0.03] = np.nan
    return s, r


@pytest.mark.skipif(not portfolio.HAVE_CPP, reason="C++ extension not built")
def test_cpp_matches_numpy():
    s, r = _random()
    a_pnl, a_to = portfolio.long_short(s, r, 0.1, use_cpp=False)
    b_pnl, b_to = portfolio.long_short(s, r, 0.1, use_cpp=True)
    np.testing.assert_allclose(a_pnl, b_pnl, atol=1e-12)
    np.testing.assert_allclose(a_to, b_to, atol=1e-12)
    np.testing.assert_allclose(
        portfolio.rank_ic(s, r, use_cpp=False), portfolio.rank_ic(s, r, use_cpp=True), atol=1e-10, equal_nan=True
    )


def test_rank_ic_matches_scipy():
    s, r = _random(T=5)
    ic = portfolio.rank_ic(s, r, use_cpp=False)
    for t in range(5):
        ok = np.isfinite(s[t]) & np.isfinite(r[t])
        assert math.isclose(ic[t], spearmanr(s[t, ok], r[t, ok]).statistic, abs_tol=1e-12)


def test_portfolio_is_dollar_neutral_and_unit_gross():
    s, _ = _random()
    # A return equal on every stock must give zero P&L for a dollar-neutral book.
    flat = np.full_like(s, 0.01)
    pnl, turnover = portfolio.long_short(s, flat, 0.1)
    np.testing.assert_allclose(pnl, 0.0, atol=1e-12)
    assert math.isclose(turnover[0], 1.0)  # first day: build a book of gross size 1


def test_perfect_foresight_gives_huge_sharpe():
    _, r = _random(T=250)
    pnl, _ = portfolio.long_short(np.nan_to_num(r, nan=0.0), r, 0.1)
    assert stats.sharpe(pnl) > 10


def test_sharpe_and_newey_west():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 5000)
    assert math.isclose(stats.sharpe(x), x.mean() / x.std(ddof=1) * math.sqrt(252))
    naive_t = x.mean() / (x.std(ddof=1) / math.sqrt(x.size))
    assert abs(stats.newey_west_tstat(x) - naive_t) / abs(naive_t) < 0.1  # iid: close to naive


def test_deflation_lowers_confidence():
    rng = np.random.default_rng(3)
    trials = rng.normal(0, 0.01, (40, 500))
    srs = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
    best = trials[np.argmax(srs)]
    assert stats.deflated_sharpe_ratio(best, srs) < stats.probabilistic_sharpe_ratio(best)
