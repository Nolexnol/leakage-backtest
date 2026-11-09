"""Point-in-time features and forward-return targets.

Timing convention (the whole project hinges on this):

* A feature value at date ``t`` may use only data observed up to the close of ``t``.
* The honest target assumes the signal is computed after the close of ``t`` and
  the trade happens at the **next open** (``t+1``), held for ``h`` days:
  ``open[t+1+h] / open[t+1] - 1``.

Every feature here is a function of past prices only, which the leakage audit in
``leakbt.audit`` checks mechanically.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

from .data import Panel

FeatureFn = Callable[[Panel], pd.DataFrame]


def _ret(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.shift(n) - 1.0


def _daily(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


# Each entry: name -> function(panel) -> wide frame (date x ticker), value known at close t.
FEATURES: Dict[str, FeatureFn] = {
    # Short-term reversal / continuation
    "ret_1": lambda p: _ret(p.close, 1),
    "ret_5": lambda p: _ret(p.close, 5),
    "ret_21": lambda p: _ret(p.close, 21),
    # 12-1 month momentum (skip the most recent month)
    "mom_252_21": lambda p: p.close.shift(21) / p.close.shift(252) - 1.0,
    # Risk and liquidity
    "vol_21": lambda p: _daily(p.close).rolling(21, min_periods=15).std(),
    "range_5": lambda p: ((p.high - p.low) / p.close).rolling(5, min_periods=3).mean(),
    "volume_z": lambda p: np.log(p.volume.replace(0, np.nan))
    - np.log(p.volume.replace(0, np.nan)).rolling(21, min_periods=15).mean(),
    # Position within recent range
    "dist_high_21": lambda p: p.close / p.close.rolling(21, min_periods=15).max() - 1.0,
}

FEATURE_NAMES = list(FEATURES)


# A deliberately buggy feature, used only to show that the audit catches it.
# ``shift(-1)`` silently pulls tomorrow's close into today's row -- a one-character
# mistake that is common in real pandas code.
def _fullsample_z(frame: pd.DataFrame) -> pd.DataFrame:
    # Standardizing each stock with its *whole-history* mean and std: the mean
    # and std include future observations, so the feature leaks.
    return (frame - frame.mean()) / frame.std()


BUGGY_FEATURES: Dict[str, FeatureFn] = {
    "ret_5_offbyone": lambda p: p.close.shift(-1) / p.close.shift(4) - 1.0,
    "vol_21_fullsample_z": lambda p: _fullsample_z(_daily(p.close).rolling(21, min_periods=15).std()),
}


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank each row (date) across tickers and rescale to [-0.5, 0.5].

    Ranking within a single date uses no information from other dates, so it
    cannot leak future data (unlike z-scoring with full-sample means).
    """
    ranks = frame.rank(axis=1, pct=True)
    return ranks - 0.5


def build_features(panel: Panel, names: list[str] | None = None, include_buggy: bool = False) -> Dict[str, pd.DataFrame]:
    """Compute raw (un-ranked) features as wide frames."""
    registry = dict(FEATURES)
    if include_buggy:
        registry.update(BUGGY_FEATURES)
    names = names or list(registry)
    return {n: registry[n](panel) for n in names}


def forward_return(panel: Panel, h: int = 1, execution: str = "next_open") -> pd.DataFrame:
    """Forward return over ``h`` days for a signal formed at the close of ``t``.

    execution="next_open": honest. Enter at open t+1, exit at open t+1+h.
    execution="same_close": look-ahead. Pretends you can trade at the very close
        price that was used to compute the signal (close t -> close t+h).
    """
    if execution == "next_open":
        return panel.open.shift(-(1 + h)) / panel.open.shift(-1) - 1.0
    if execution == "same_close":
        return panel.close.shift(-h) / panel.close - 1.0
    raise ValueError(f"unknown execution mode: {execution}")


def demean(frame: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional mean each day (market-neutral target)."""
    return frame.sub(frame.mean(axis=1), axis=0)


def to_long(features: Dict[str, pd.DataFrame], target: pd.DataFrame) -> pd.DataFrame:
    """Stack wide frames into a long (date, ticker) table of ranked features + target.

    Rows with any missing feature or a missing target are dropped.
    """
    ranked = {n: cross_sectional_rank(f).stack(future_stack=True) for n, f in features.items()}
    df = pd.DataFrame(ranked)
    df["target"] = target.stack(future_stack=True)
    df.index.names = ["date", "ticker"]
    return df.dropna()
