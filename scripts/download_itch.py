"""Download NASDAQ's TotalView-ITCH 5.0 sample days and convert them for reports 07 and 08.

NASDAQ publishes a few full days of its ITCH feed at emi.nasdaq.com/ITCH/Nasdaq ITCH/ (3.5-6 GB
each, gzipped). The 2018 days listed in older guides are gone (only their .md5sum files remain),
and the server does not serve .md5sum files at all, so each download is checked against the
advertised size; gzip's own CRC is checked when the converter reads the file to the end.

For each day the script downloads (resuming a partial file), then runs `markout.lob.itch` for
the ten stocks of report 07. Raw files stay in data/raw/itch/ and are never committed.

Usage:  .venv/bin/python scripts/download_itch.py [--days 01302019 ...] [--keep-going]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

from markout.lob import itch, lobster
from markout.lob.multiday import TICKERS

BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
DAYS = (
    "01302019.NASDAQ_ITCH50.gz",
    "03272019.NASDAQ_ITCH50.gz",
    "07302019.NASDAQ_ITCH50.gz",
    "08302019.NASDAQ_ITCH50.gz",
    "S101819-v50.txt.gz",
    "10302019.NASDAQ_ITCH50.gz",
    "12302019.NASDAQ_ITCH50.gz",
    "01302020.NASDAQ_ITCH50.gz",
    "S071321-v50.txt.gz",
    "S081321-v50.txt.gz",
)


def remote_size(name: str) -> int:
    req = urllib.request.Request(BASE + name, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(name: str, dest: Path) -> Path:
    """curl resumes a partial file and retries; the result must have the advertised size."""
    want = remote_size(name)
    if dest.exists() and dest.stat().st_size == want:
        return dest
    for _ in range(6):
        subprocess.run(["curl", "-sS", "--retry", "5", "-C", "-", "-o", str(dest), BASE + name], check=False)
        if dest.exists() and dest.stat().st_size == want:
            return dest
    raise OSError(f"{name}: got {dest.stat().st_size if dest.exists() else 0} of {want} bytes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", nargs="*", help="file names or their MMDDYYYY prefixes (default: all ten)")
    ap.add_argument("--keep-going", action="store_true", help="skip a day that fails instead of stopping")
    a = ap.parse_args(argv)
    names = [n for n in DAYS if not a.days or any(n.startswith(d) for d in a.days)]
    itch.ITCH_RAW.mkdir(parents=True, exist_ok=True)
    for name in names:
        day = str(itch.file_date(Path(name)))
        if lobster.available(TICKERS, day):
            print(f"{day}: already converted")
            continue
        try:
            path = download(name, itch.ITCH_RAW / name)
            r = itch.convert(path, list(TICKERS))
        except OSError as e:
            if not a.keep_going:
                raise
            print(f"{day}: skipped ({e})")
            continue
        bad = sum(sum(s["unmatched"].values()) for s in r["tickers"].values())
        print(f"{day}: {r['itch_messages']:,} messages, {bad} unmatched references", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
