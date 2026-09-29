"""House chart style: one palette, a light and a dark variant of every figure.

Every figure is drawn by a zero-argument function that reads colors through this
module, so `render` can call it once per theme:

    from markout import plotting as mp

    def draw():
        fig, ax = mp.subplots()
        ax.plot(x, model, color=mp.series(0), label="LightGBM")
        ax.plot(x, base, color=mp.series(1), label="median baseline")
        mp.legend(ax)
        mp.title(ax, "Out-of-sample MAE by day", "bps, lower is better")
        return fig

    mp.render("a_mae_by_day", draw)   # reports/figures/a_mae_by_day.png + _dark.png

Rules (from the dataviz method; the palette is its validated default instance):
- categorical colors in fixed slot order, at most 8 series, never cycled;
- one y-axis per plot, never twinx: use two panels instead;
- text wears ink colors, never series colors;
- hairline solid y-grid, no top/right/left spines, a baseline at the bottom;
- 1.5 pt lines (~2 px), 6 pt markers (~8 px) with a 1.5 pt surface-colored ring;
- sequential = one hue (blue); diverging = red <- gray -> blue, gray at zero;
- a legend whenever there are >= 2 series; direct labels only where they help.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from markout.paths import FIGURES  # noqa: E402

LIGHT = {
    "name": "light",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "div": ("#e34948", "#f0efec", "#2a78d6"),  # negative, zero, positive
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
}

DARK = {
    "name": "dark",
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "baseline": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"],
    "div": ("#e66767", "#383835", "#3987e5"),
    # low values recede into the dark surface, so the ramp runs dark -> light
    "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
}

_theme: dict = LIGHT


def theme() -> dict:
    return _theme


def series(i: int) -> str:
    """Categorical color for slot i (0-based)."""
    if not 0 <= i < 8:
        raise ValueError("at most 8 categorical series; fold the rest into 'Other' or facet")
    return _theme["series"][i]


def ink(role: str = "primary") -> str:
    return {"primary": _theme["ink"], "secondary": _theme["ink2"], "muted": _theme["muted"]}[role]


def surface() -> str:
    return _theme["surface"]


def baseline_color() -> str:
    return _theme["baseline"]


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"seq_{_theme['name']}", _theme["seq"])


def diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"div_{_theme['name']}", list(_theme["div"]))


def diverging_norm(vmin: float, vmax: float) -> TwoSlopeNorm:
    """Norm that pins zero to the gray midpoint (vmin < 0 < vmax)."""
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def _rc(t: dict) -> dict:
    return {
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "axes.labelsize": 9.5,
        "axes.titlecolor": t["ink"],
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": t["baseline"],
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=t["series"]),
        "grid.color": t["grid"],
        "grid.linewidth": 0.75,
        "grid.linestyle": "-",
        "xtick.color": t["baseline"],
        "ytick.color": t["baseline"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": t["surface"],
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": t["ink2"],
        "legend.handlelength": 1.6,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }


@contextmanager
def use(t: dict) -> Iterator[dict]:
    """Temporarily switch the active theme (colors and rcParams)."""
    global _theme
    previous = _theme
    _theme = t
    try:
        with plt.rc_context(_rc(t)):
            yield t
    finally:
        _theme = previous


def subplots(nrows: int = 1, ncols: int = 1, w: float = 7.2, h: float = 3.8, **kw):
    """plt.subplots with the house default size (720 x 380 px at 100 dpi)."""
    kw.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def title(ax, text: str, subtitle: str | None = None) -> None:
    """Left-aligned title, with an optional secondary-ink subtitle under it."""
    if subtitle:
        ax.set_title(text, loc="left", pad=24)
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9, color=ink("secondary"))
    else:
        ax.set_title(text, loc="left")


def legend(ax, **kw) -> None:
    kw.setdefault("loc", "best")
    ax.legend(**kw)


def zero_line(ax, y: float = 0.0) -> None:
    ax.axhline(y, color=baseline_color(), linewidth=0.9, zorder=1)


def end_label(ax, x: float, y: float, text: str, dx: float = 5, dy: float = 0, **kw) -> None:
    """Direct label at a line's end, in secondary ink (never the series color)."""
    kw.setdefault("fontsize", 9)
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha="left", va="center", color=ink("secondary"), **kw)


def bars(ax, x, heights, width: float = 0.62, color: str | None = None,
         horizontal: bool = False, **kw):
    """Bars with a surface-colored gap between neighbours (no drawn borders)."""
    color = color or series(0)
    style = dict(color=color, edgecolor=surface(), linewidth=1.5, zorder=2, **kw)
    if horizontal:
        return ax.barh(x, heights, height=width, **style)
    return ax.bar(x, heights, width=width, **style)


def render(name: str, draw: Callable[[], "plt.Figure"], dark: bool = True) -> list[Path]:
    """Call `draw` under each theme and save reports/figures/<name>[_dark].png."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    variants = [(LIGHT, "")] + ([(DARK, "_dark")] if dark else [])
    paths = []
    for t, suffix in variants:
        with use(t):
            fig = draw()
            path = FIGURES / f"{name}{suffix}.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def picture(name: str, alt: str, prefix: str = "figures/", width: int = 720) -> str:
    """Markdown/HTML snippet that shows the dark variant to dark-mode readers."""
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{prefix}{name}_dark.png">\n'
        f'  <img alt="{alt}" src="{prefix}{name}.png" width="{width}">\n'
        "</picture>"
    )
