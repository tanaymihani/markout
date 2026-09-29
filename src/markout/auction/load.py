"""Optiver "Trading at the Close" (Kaggle): download, convert to Parquet, load.

The data is the NASDAQ closing auction: 200 stocks, date_id 0-480, a snapshot every
10 s from 0 to 540 s into the 10-minute auction. Prices are normalized to each
stock's WAP at the start of the auction, sizes are in USD, and `target` is the
stock's 60 s WAP move minus the synthetic index's 60 s move, in bps.

The raw CSV (~640 MB) is converted once to zstd Parquet with float32 values and
then deleted: this project was built on a machine with 8 GB of RAM and little
free disk. The data cannot be redistributed (Kaggle competition rules), so it is
never committed; `python -m markout.auction.load` re-downloads it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import polars as pl

from markout.paths import PROCESSED, RAW

COMPETITION = "optiver-trading-at-the-close"
RAW_DIR = RAW / "optiver"
OUT_DIR = PROCESSED / "optiver"
PARQUET = OUT_DIR / "train.parquet"

SCHEMA: dict[str, pl.DataType] = {
    "stock_id": pl.Int16,
    "date_id": pl.Int16,
    "seconds_in_bucket": pl.Int16,
    "imbalance_size": pl.Float32,
    "imbalance_buy_sell_flag": pl.Int8,
    "reference_price": pl.Float32,
    "matched_size": pl.Float32,
    "far_price": pl.Float32,
    "near_price": pl.Float32,
    "bid_price": pl.Float32,
    "bid_size": pl.Float32,
    "ask_price": pl.Float32,
    "ask_size": pl.Float32,
    "wap": pl.Float32,
    "target": pl.Float32,
}
KEYS = ["stock_id", "date_id", "seconds_in_bucket"]


def download(dest: Path = RAW_DIR) -> Path:
    """Fetch train.csv with the Kaggle CLI (needs `kaggle auth login` and the
    competition rules accepted on kaggle.com). Returns the CSV path."""
    dest.mkdir(parents=True, exist_ok=True)
    kaggle = Path(sys.executable).with_name("kaggle")
    cmd = [str(kaggle), "competitions", "download", "-c", COMPETITION, "-f", "train.csv", "-p", str(dest)]
    subprocess.run(cmd, check=True)
    csv = dest / "train.csv"
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()
    if not csv.exists():
        raise FileNotFoundError(f"download finished but {csv} is missing")
    return csv


def to_parquet(csv: Path, out: Path = PARQUET, delete_csv: bool = True) -> Path:
    """CSV -> zstd Parquet with compact dtypes, sorted by stock, date, second."""
    out.parent.mkdir(parents=True, exist_ok=True)
    df = (
        pl.scan_csv(csv, schema_overrides=SCHEMA)
        .select(list(SCHEMA))
        .with_columns([pl.col(c).cast(t) for c, t in SCHEMA.items()])
        .sort(KEYS)
        .collect()
    )
    df.write_parquet(out, compression="zstd", statistics=True)
    if delete_csv:
        csv.unlink()
    return out


def load(columns: list[str] | None = None, path: Path = PARQUET) -> pl.DataFrame:
    """Load the converted training data (sorted by stock, date, second)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m markout.auction.load` first")
    return pl.read_parquet(path, columns=columns)


def available(path: Path = PARQUET) -> bool:
    return path.exists()


def main() -> None:
    if PARQUET.exists():
        print(f"already converted: {PARQUET}")
        return
    csv = RAW_DIR / "train.csv"
    if not csv.exists():
        csv = download()
    out = to_parquet(csv)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    if RAW_DIR.exists() and not any(RAW_DIR.iterdir()):
        shutil.rmtree(RAW_DIR)


if __name__ == "__main__":
    main()
