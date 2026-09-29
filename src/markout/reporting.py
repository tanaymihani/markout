"""Helpers for writing the markdown reports. Numbers are always interpolated from
computed results, never typed by hand."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from markout.paths import REPORTS


def num(x: float | int | None, digits: int = 3, signed: bool = False, pct: bool = False) -> str:
    """Format a number for prose: 3 significant-ish digits, thousands commas."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if pct:
        x = 100.0 * x
    sign = "+" if signed and x > 0 else ""
    ax = abs(x)
    if isinstance(x, int) or (ax >= 1000 and float(x).is_integer()):
        s = f"{int(round(x)):,}"
    elif ax >= 100:
        s = f"{x:,.0f}"
    elif ax >= 10:
        s = f"{x:.1f}"
    elif ax >= 1:
        s = f"{x:.{max(digits - 1, 0)}f}"
    elif ax == 0:
        s = "0"
    else:
        s = f"{x:.{digits}g}"
    return f"{sign}{s}{'%' if pct else ''}"


def prob(p: float | None) -> str:
    """Probabilities and p-values: never print 0.99999 as "1" or 1e-48 as a long float."""
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    if p > 0.999:
        return "> 0.999"
    return f"{p:.3f}"


def usd(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def table(rows: Sequence[Mapping], columns: Sequence[str] | None = None,
          formats: Mapping[str, str] | None = None, headers: Mapping[str, str] | None = None) -> str:
    """GitHub-flavoured markdown table from a list of dicts."""
    if not rows:
        return "_(no rows)_"
    columns = list(columns or rows[0].keys())
    formats = dict(formats or {})
    headers = dict(headers or {})

    def cell(col: str, v) -> str:
        if v is None:
            return "n/a"
        if col in formats and isinstance(v, (int, float)):
            try:
                return formats[col].format(v)
            except (ValueError, TypeError):
                return str(v)
        if isinstance(v, float):
            return num(v)
        return str(v)

    head = "| " + " | ".join(headers.get(c, c) for c in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(cell(c, r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {s}" for s in items)


def write(name: str, text: str) -> Path:
    """Write reports/<name>.md."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{name}.md"
    path.write_text(text.rstrip() + "\n")
    return path
