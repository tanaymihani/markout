"""Daily S&P 500, VIX and 3-month VIX closes from Yahoo Finance (via yfinance, free).

Saved once to data/raw/vol/ (never committed: Yahoo's data is for personal use) and
aligned on trading days. The report reads the saved files and never refetches, so it is
reproducible from the snapshot.
"""

from __future__ import annotations

import pandas as pd

from markout.paths import RAW

DIR = RAW / "vol"
TICKERS = {"spx": "^GSPC", "vix": "^VIX", "vix3m": "^VIX3M"}


def download(start: str = "1990-01-01") -> dict[str, int]:
    import yfinance as yf

    DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, tk in TICKERS.items():
        h = yf.Ticker(tk).history(start=start, auto_adjust=False)
        s = h["Close"].rename(name)
        s.index = pd.to_datetime(s.index.date)
        s.to_frame().to_parquet(DIR / f"{name}.parquet")
        counts[name] = len(s)
    return counts


def available() -> bool:
    return all((DIR / f"{n}.parquet").exists() for n in TICKERS)


def load() -> pd.DataFrame:
    """One row per S&P trading day: spx close, vix, vix3m (NaN before July 2006)."""
    frames = [pd.read_parquet(DIR / f"{n}.parquet") for n in TICKERS]
    df = frames[0].join(frames[1], how="inner").join(frames[2], how="left")
    return df.sort_index()


if __name__ == "__main__":
    print(download())
