"""Mechanical leakage audit: the truncation test.

Idea: if a feature is truly point-in-time, its value at date ``t`` cannot
change when you delete every row after ``t``. Recompute each feature on the
truncated panel at a sample of dates and compare with the full-sample values.
Any mismatch means the feature used future data.

This catches off-by-one ``shift(-1)`` bugs, centered rolling windows, full-sample
normalization and similar mistakes without having to read the code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

from .data import Panel


@dataclass
class AuditResult:
    feature: str
    dates_checked: int
    mismatched_dates: int
    max_abs_diff: float

    @property
    def leaks(self) -> bool:
        return self.mismatched_dates > 0


def truncation_audit(
    panel: Panel,
    features: Dict[str, Callable[[Panel], pd.DataFrame]],
    n_dates: int = 12,
    seed: int = 0,
    atol: float = 1e-12,
) -> list[AuditResult]:
    rng = np.random.default_rng(seed)
    dates = panel.dates
    # Skip the warm-up period and the last few rows (nothing after them to leak).
    candidates = dates[300:-10]
    sample = sorted(rng.choice(len(candidates), size=min(n_dates, len(candidates)), replace=False))
    check_dates = [candidates[i] for i in sample]

    full = {name: fn(panel) for name, fn in features.items()}
    results = []
    for name, fn in features.items():
        mismatches, worst = 0, 0.0
        for t in check_dates:
            truncated_value = fn(panel.truncate(t)).loc[t]
            full_value = full[name].loc[t]
            a, b = full_value.to_numpy(), truncated_value.reindex(full_value.index).to_numpy()
            both_nan = np.isnan(a) & np.isnan(b)
            diff = np.where(both_nan, 0.0, np.abs(a - b))
            diff = np.nan_to_num(diff, nan=np.inf)  # NaN on one side only = mismatch
            if (diff > atol).any():
                mismatches += 1
                worst = max(worst, float(np.max(diff[np.isfinite(diff)], initial=0.0)) if np.isfinite(diff).any() else np.inf)
        results.append(AuditResult(name, len(check_dates), mismatches, worst))
    return results


def format_audit(results: Iterable[AuditResult]) -> str:
    lines = ["| Feature | Dates checked | Dates where value changed | Verdict |", "| --- | --- | --- | --- |"]
    for r in results:
        verdict = "LEAKS future data" if r.leaks else "point-in-time"
        lines.append(f"| `{r.feature}` | {r.dates_checked} | {r.mismatched_dates} | {verdict} |")
    return "\n".join(lines)
