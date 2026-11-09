import numpy as np
import pandas as pd

from leakbt.audit import truncation_audit
from leakbt.features import BUGGY_FEATURES, FEATURES, cross_sectional_rank, forward_return


def test_all_production_features_are_point_in_time(panel):
    results = truncation_audit(panel, FEATURES, n_dates=8)
    leaking = [r.feature for r in results if r.leaks]
    assert leaking == []


def test_audit_catches_each_planted_bug(panel):
    results = {r.feature: r for r in truncation_audit(panel, BUGGY_FEATURES, n_dates=8)}
    for name in BUGGY_FEATURES:
        assert results[name].leaks, f"audit missed {name}"


def test_next_open_target_starts_after_signal_date(panel):
    h = 3
    tgt = forward_return(panel, h, "next_open")
    t = panel.dates[50]
    expected = panel.open.loc[panel.dates[51 + h]] / panel.open.loc[panel.dates[51]] - 1
    pd.testing.assert_series_equal(tgt.loc[t], expected, check_names=False)


def test_cross_sectional_rank_is_bounded_and_per_day(panel):
    r = cross_sectional_rank(panel.close)
    assert r.max().max() <= 0.5 + 1e-12 and r.min().min() >= -0.5
    # Changing a later day must not change an earlier day's ranks.
    altered = panel.close.copy()
    altered.iloc[-1] = altered.iloc[-1][::-1].to_numpy()
    np.testing.assert_allclose(cross_sectional_rank(altered).iloc[:-1], r.iloc[:-1])
