import pandas as pd
import pytest

from leakbt.validation import blocked_kfold, purged_walk_forward, shuffled_kfold_rows


@pytest.mark.parametrize("horizon,embargo", [(1, 0), (5, 5), (10, 2)])
def test_purged_walk_forward_leaves_a_gap(horizon, embargo):
    dates = pd.bdate_range("2015-01-01", periods=600)
    pos = {d: i for i, d in enumerate(dates)}
    splits = list(purged_walk_forward(dates, horizon, min_train=200, test_size=21, embargo=embargo))
    assert splits
    for train, test in splits:
        last_train, first_test = pos[train[-1]], pos[test[0]]
        # A training target at s uses prices through s + 1 + horizon; it must end before testing starts.
        assert last_train + 1 + horizon + embargo <= first_test
        assert train.max() < test.min()


def test_walk_forward_covers_every_test_day_once():
    dates = pd.bdate_range("2015-01-01", periods=500)
    tests = [t for _, t in purged_walk_forward(dates, 1, min_train=100, test_size=21)]
    covered = pd.DatetimeIndex(sorted(d for t in tests for d in t))
    assert covered.equals(dates[100:])


def test_kfolds_partition_the_data():
    folds = list(shuffled_kfold_rows(103, k=5))
    test_rows = sorted(i for _, te in folds for i in te)
    assert test_rows == list(range(103))
    dates = pd.bdate_range("2015-01-01", periods=50)
    assert sum(len(te) for _, te in blocked_kfold(dates, 5)) == 50
