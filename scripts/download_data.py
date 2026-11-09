"""Download the daily S&P 500 stock panel used in this project.

Source: plotly/datasets on GitHub (a mirror of the Kaggle "S&P 500 stock data"
set): daily OHLCV for 505 S&P 500 constituents, Feb 2013 - Feb 2018.

Usage:
    python scripts/download_data.py            # saves to data/all_stocks_5yr.csv
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "all_stocks_5yr.csv"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already downloaded: {DEST}")
        return 0
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved {DEST.stat().st_size / 1e6:.1f} MB to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
