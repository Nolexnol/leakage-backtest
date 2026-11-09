import numpy as np
import pandas as pd
import pytest

from leakbt.data import Panel


@pytest.fixture(scope="session")
def panel() -> Panel:
    """Small synthetic price panel: 400 business days x 30 tickers."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", periods=400)
    tickers = [f"S{i:02d}" for i in range(30)]
    rets = rng.normal(0, 0.015, size=(len(dates), len(tickers)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, close.shape)))
    volume = pd.DataFrame(rng.lognormal(14, 0.4, close.shape), index=dates, columns=tickers)
    return Panel(open=open_, high=high, low=low, close=close, volume=volume)
