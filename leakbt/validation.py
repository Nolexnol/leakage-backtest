"""Train/test splitting schemes, from leaky to honest.

All splitters yield ``(train_dates, test_dates)`` pairs of DatetimeIndex objects.
Splitting by *date* (never by row) keeps a whole cross-section on one side.

* ``shuffled_kfold``  -- rows from all periods mixed at random. Neighbouring days
  share most of an ``h``-day forward return, so the model is tested on
  near-copies of its training data. This is the leak.
* ``blocked_kfold``   -- contiguous date blocks, but the model still trains on
  data from *after* the test block and nothing is purged at the boundaries.
* ``purged_walk_forward`` -- train only on the past, drop (purge) the last
  ``h`` days of training whose targets overlap the test window, and leave an
  extra ``embargo`` gap. This is the honest scheme.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

Split = Tuple[pd.DatetimeIndex, pd.DatetimeIndex]


def shuffled_kfold_rows(n_rows: int, k: int = 5, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Row-level shuffled K-fold (what ``sklearn.model_selection.KFold(shuffle=True)`` does)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    for fold in np.array_split(order, k):
        mask = np.zeros(n_rows, dtype=bool)
        mask[fold] = True
        yield np.flatnonzero(~mask), np.flatnonzero(mask)


def blocked_kfold(dates: pd.DatetimeIndex, k: int = 5) -> Iterator[Split]:
    for block in np.array_split(np.arange(len(dates)), k):
        test = dates[block]
        train = dates.difference(test)
        yield train, test


def purged_walk_forward(
    dates: pd.DatetimeIndex,
    horizon: int,
    min_train: int = 504,
    test_size: int = 21,
    embargo: int = 5,
    max_train: int | None = None,
) -> Iterator[Split]:
    """Expanding (or rolling, if ``max_train``) walk-forward with purging.

    A training row at date ``s`` has a target that uses prices through
    ``s + 1 + horizon`` (next-open execution). It is only usable for a test
    window starting at index ``i`` if ``s + 1 + horizon + embargo <= i``.
    """
    n = len(dates)
    gap = horizon + 1 + embargo
    start = min_train
    while start < n:
        stop = min(start + test_size, n)
        train_end = start - gap  # exclusive
        if train_end <= 0:
            start = stop
            continue
        train_start = 0 if max_train is None else max(0, train_end - max_train)
        yield dates[train_start:train_end], dates[start:stop]
        start = stop
