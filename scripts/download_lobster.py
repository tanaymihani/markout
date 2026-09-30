"""Download the LOBSTER level-10 sample files and convert them to zstd Parquet.

Five tickers (AAPL, AMZN, GOOG, INTC, MSFT) on 2012-06-21, 09:30-16:00, 10 levels.

Where the files come from
-------------------------
The official links (``https://lobsterdata.com/info/sample/LOBSTER_SampleFile_<T>_2012-06-21_10.zip``,
confirmed from an archived copy of https://lobsterdata.com/info/DataSamples.php) stopped
working when lobsterdata.com was rebuilt as a single-page app: every path now answers
``200 text/html`` with the app shell, and the Wayback Machine never stored the zips. The
script still tries the official URL first and accepts the response only if it really is a
zip archive. Otherwise it falls back to a public Hugging Face mirror of the same
extracted CSVs (``totalorganfailure/lobster-data``), pinned to one commit, and checks every
file against the size and hash that mirror publishes (sha256 for Git-LFS files, the git
blob SHA-1 for small files). The hashes prove the transfer is intact, not that the
mirror is authentic. Authenticity is checked by the data themselves:
``markout.lob.lobster.check_consistency`` replays every message against the order book,
which a corrupted or edited file would fail.

For each ticker the script downloads, verifies, converts to
``data/processed/lobster/<T>_messages.parquet`` and ``<T>_book.parquet``, and deletes the CSVs
before moving on to the next ticker, so at most one ticker's CSVs are ever on disk.

Usage:  .venv/bin/python scripts/download_lobster.py [--tickers AAPL MSFT] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from markout.lob.lobster import (
    BOOK_COLUMNS,
    DATE,
    LEVELS,
    MESSAGE_COLUMNS,
    TICKERS,
    book_path,
    message_path,
)
from markout.paths import PROCESSED, RAW

RAW_DIR = RAW / "lobster"
OUT_DIR = PROCESSED / "lobster"
OFFICIAL = "https://lobsterdata.com/info/sample/LOBSTER_SampleFile_{t}_{d}_{lvl}.zip"
MIRROR_REPO = "totalorganfailure/lobster-data"
MIRROR_REV = "fb51a829d2b5a78c79468db533ba28c5f1d161d0"  # pinned commit (2025-10-15)
MIRROR = f"https://huggingface.co/datasets/{MIRROR_REPO}/resolve/{MIRROR_REV}/{{path}}"
MIN_FREE_GB_TO_KEEP_ZIP = 3.0


@dataclass(frozen=True)
class Remote:
    """One file on the mirror: its path, size in bytes and published hash."""

    path: str
    size: int
    sha256: str | None = None      # Git-LFS files publish the sha256 of the content
    git_sha1: str | None = None    # small files publish the git blob id instead


def _stem(t: str) -> str:
    return f"LOBSTER_SampleFile_{t}_{DATE}_{LEVELS}/{t}_{DATE}_34200000_57600000"


MIRROR_FILES: dict[str, tuple[Remote, Remote]] = {
    "AAPL": (
        Remote(f"{_stem('AAPL')}_message_10.csv", 16641275,
               sha256="6562394b996138d5c4527b1282e77a1a385e844110550c34628fbd73bc5411e5"),
        Remote(f"{_stem('AAPL')}_orderbook_10.csv", 93481244,
               sha256="ed75450031996e81bdc5fc985f6afe6851c81d7b2b39bccf13c56d845fe6ff5d"),
    ),
    "AMZN": (
        Remote(f"{_stem('AMZN')}_message_10.csv", 11248750,
               sha256="c85a75b51f3616a683f825c2e03f984535e9329a8be5ab12325130d32223a2c5"),
        Remote(f"{_stem('AMZN')}_orderbook_10.csv", 63216242,
               sha256="decfe952b3fa06c9922dc8fba763a37a0fb032d4231767997e4abf2aa0d29b50"),
    ),
    "GOOG": (
        Remote(f"{_stem('GOOG')}_message_10.csv", 6094817,
               git_sha1="dd7f19715f5fd76cb4d52240c38c1a274d8e74a9"),
        Remote(f"{_stem('GOOG')}_orderbook_10.csv", 34073243,
               sha256="9ebe00b4f1aee57aa73dcb979e1074d16554296ca32b87440ce756c60974bb53"),
    ),
    "INTC": (
        Remote(f"{_stem('INTC')}_message_10.csv", 25663091,
               sha256="199c55b35f6c62f07b058922038c0b8d168bc3be9d2de94da19c901cd3f29927"),
        Remote(f"{_stem('INTC')}_orderbook_10.csv", 158466105,
               sha256="6bc4c4dc1e92d78e46d04f9e7914df2d8c2a9231928493e7395a9085e1fc91ab"),
    ),
    "MSFT": (
        Remote(f"{_stem('MSFT')}_message_10.csv", 27582358,
               sha256="669e5c35b1eab0f7336fa9ed919002e65b08d2aabd96bb4ae88d647184e7a18d"),
        Remote(f"{_stem('MSFT')}_orderbook_10.csv", 170192771,
               sha256="ec2f53467cf7bfb00b5702381bea7b4a061bb4c1730454df3679b272b3561c0e"),
    ),
}
README = Remote(f"LOBSTER_SampleFile_GOOG_{DATE}_{LEVELS}/LOBSTER_SampleFiles_ReadMe.txt", 4790,
                git_sha1="23c806b08ae7646a96c12b98c2dd5950bb36d69b")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def fetch(url: str, dest: Path, timeout: float = 120.0) -> Path:
    """Stream `url` to `dest` in 1 MB chunks (never holds a whole file in memory)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "markout-research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)
    return dest


def verify(path: Path, remote: Remote) -> None:
    """Check size and the mirror's published hash; raise on any mismatch."""
    size = path.stat().st_size
    if size != remote.size:
        raise ValueError(f"{path.name}: size {size} != expected {remote.size}")
    if remote.sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != remote.sha256:
            raise ValueError(f"{path.name}: sha256 mismatch")
    if remote.git_sha1:
        h = hashlib.sha1(f"blob {size}\0".encode())
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != remote.git_sha1:
            raise ValueError(f"{path.name}: git blob sha1 mismatch")


def try_official(ticker: str, workdir: Path) -> tuple[Path, Path] | None:
    """Download and unzip the official sample if the URL still serves a real zip."""
    url = OFFICIAL.format(t=ticker, d=DATE, lvl=LEVELS)
    zpath = RAW_DIR / Path(url).name
    try:
        if not zpath.exists():
            fetch(url, zpath)
        with open(zpath, "rb") as f:
            if f.read(2) != b"PK":      # the rebuilt site answers every path with HTML
                zpath.unlink()
                return None
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            msg = next(n for n in names if n.endswith(f"_message_{LEVELS}.csv"))
            ob = next(n for n in names if n.endswith(f"_orderbook_{LEVELS}.csv"))
            z.extract(msg, workdir)
            z.extract(ob, workdir)
        if free_gb(RAW_DIR) < MIN_FREE_GB_TO_KEEP_ZIP:
            zpath.unlink()
        return workdir / msg, workdir / ob
    except (urllib.error.URLError, OSError, zipfile.BadZipFile, StopIteration):
        if zpath.exists():
            zpath.unlink()
        return None


def from_mirror(ticker: str, workdir: Path) -> tuple[Path, Path]:
    paths = []
    for remote in MIRROR_FILES[ticker]:
        dest = workdir / Path(remote.path).name
        if not dest.exists():
            fetch(MIRROR.format(path=remote.path), dest)
        verify(dest, remote)
        paths.append(dest)
    return paths[0], paths[1]


def convert(msg_csv: Path, book_csv: Path, ticker: str) -> tuple[int, int]:
    """CSV -> zstd Parquet with a fixed schema. Prices stay int64 (dollars x 10^4)."""
    msg_schema = {"time": pl.Float64, "type": pl.Int8, "order_id": pl.Int64,
                  "size": pl.Int64, "price": pl.Int64, "direction": pl.Int8}
    messages = pl.read_csv(msg_csv, has_header=False, new_columns=list(MESSAGE_COLUMNS),
                           schema_overrides=msg_schema)
    if messages.columns != list(MESSAGE_COLUMNS):
        raise ValueError(f"{ticker}: unexpected message columns {messages.columns}")
    messages.write_parquet(message_path(ticker), compression="zstd", compression_level=9)
    n_msg = messages.height
    del messages

    book = pl.read_csv(book_csv, has_header=False, new_columns=list(BOOK_COLUMNS),
                       schema_overrides={c: pl.Int64 for c in BOOK_COLUMNS})
    if book.width != 4 * LEVELS:
        raise ValueError(f"{ticker}: expected {4 * LEVELS} book columns, got {book.width}")
    book.write_parquet(book_path(ticker), compression="zstd", compression_level=9)
    n_book = book.height
    del book
    if n_msg != n_book:
        raise ValueError(f"{ticker}: {n_msg} messages but {n_book} book rows")
    return n_msg, n_book


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", nargs="*", default=list(TICKERS))
    ap.add_argument("--force", action="store_true", help="re-download even if Parquet exists")
    args = ap.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    readme = RAW_DIR / "LOBSTER_SampleFiles_ReadMe.txt"
    if not readme.exists():
        fetch(MIRROR.format(path=README.path), readme)
        verify(readme, README)

    for t in args.tickers:
        if message_path(t).exists() and book_path(t).exists() and not args.force:
            print(f"{t}: Parquet already present, skipping")
            continue
        work = RAW_DIR / f"_extract_{t}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            files = try_official(t, work)
            source = "official zip" if files else f"mirror {MIRROR_REPO}@{MIRROR_REV[:7]}"
            if files is None:
                files = from_mirror(t, work)
            n_msg, _ = convert(*files, ticker=t)
        finally:
            shutil.rmtree(work, ignore_errors=True)   # never keep extracted CSVs
        mb = (message_path(t).stat().st_size + book_path(t).stat().st_size) / 1e6
        print(f"{t}: {n_msg:,} messages from {source} -> {mb:.1f} MB Parquet; "
              f"{free_gb(OUT_DIR):.1f} GB free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
