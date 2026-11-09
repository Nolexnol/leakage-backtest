"""Load the daily stock panel into wide (date x ticker) frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "all_stocks_5yr.csv"

# Daily moves larger than this are spin-offs, unadjusted corporate actions or
# data errors in this dataset (e.g. EBAY on the PayPal spin-off, NI on the
# Columbia Pipeline spin-off, a one-day -49%/+101% glitch in LNT). We drop them
# rather than let a single bad print dominate a long-short portfolio.
MAX_ABS_DAILY_RETURN = 0.40


@dataclass(frozen=True)
class Panel:
    """Wide price panels, all indexed by trading date with one column per ticker."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    def truncate(self, last_date: pd.Timestamp) -> "Panel":
        """Return a copy containing only data up to and including ``last_date``.

        Used by the leakage audit: a feature value at date t must not change
        when every row after t is deleted.
        """
        sl = slice(None, last_date)
        return Panel(*(getattr(self, f).loc[sl].copy() for f in ("open", "high", "low", "close", "volume")))


def load_panel(path: str | Path = DEFAULT_PATH, min_history: int = 1000) -> Panel:
    """Read the long-format CSV and pivot it to wide panels.

    Tickers with fewer than ``min_history`` days are dropped so the universe is
    stable through time.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    raw = pd.read_csv(path, parse_dates=["date"])
    counts = raw.groupby("Name").size()
    keep = counts[counts >= min_history].index
    raw = raw[raw["Name"].isin(keep)]

    def wide(col: str) -> pd.DataFrame:
        return raw.pivot(index="date", columns="Name", values=col).sort_index().astype(float)

    close, open_ = wide("close"), wide("open")
    # Mask the bad prints described above in both close and open series.
    for frame in (close, open_):
        r = frame.pct_change(fill_method=None)
        bad = r.abs() > MAX_ABS_DAILY_RETURN
        # Drop the bad day and the day before it so neither side of the jump survives.
        frame[bad | bad.shift(-1, fill_value=False)] = np.nan
    return Panel(open=open_, high=wide("high"), low=wide("low"), close=close, volume=wide("volume"))
