"""Record the site's demo video: `python scripts/record_demo.py`.

Serves docs/ locally, drives the page in Chrome with Playwright (a scripted walkthrough of
every lab), and writes docs/media/markout-demo.mp4, a poster frame, and a short GIF for the
README. Needs `pip install playwright imageio-ffmpeg`; it uses the installed Google Chrome,
or Playwright's own Chromium if Chrome is missing (`playwright install chromium`).
"""

from __future__ import annotations

import functools
import http.server
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS, MEDIA = ROOT / "docs", ROOT / "docs" / "media"
SIZE = {"width": 1280, "height": 720}

# a visible cursor for the recording (headless video has none)
CURSOR = """
const c = document.createElement('div');
c.style.cssText = 'position:fixed;z-index:9999;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;' +
  'background:rgba(42,120,214,.35);border:2px solid #2a78d6;pointer-events:none;transition:transform .08s;left:-40px;top:-40px';
document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; });
document.addEventListener('mousedown', () => { c.style.transform = 'scale(.7)'; });
document.addEventListener('mouseup', () => { c.style.transform = 'scale(1)'; });
window.addEventListener('DOMContentLoaded', () => document.body.appendChild(c));
"""


def serve() -> tuple[socketserver.TCPServer, str]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # keep the console clean
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(DOCS)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/index.html"


def walkthrough(page, url: str, started: float) -> dict:
    """`started` is when the recording began (time.monotonic()), so marks are video timestamps."""
    marks = {}
    now = lambda: time.monotonic() - started  # noqa: E731

    def wait(ms):
        page.wait_for_timeout(ms)

    def scroll_to(selector, offset=70):
        page.evaluate(f"""(() => {{ const y = document.querySelector('{selector}').getBoundingClientRect().top + scrollY - {offset};
            scrollTo({{top: y, behavior: 'smooth'}}); }})()""")
        wait(1400)

    def move_to(selector):
        box = page.locator(selector).first.bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=18)
        wait(250)

    def click(selector):
        move_to(selector)
        page.locator(selector).first.click()
        wait(350)

    def slide(selector, values, pause=90):
        move_to(selector)
        for v in values:
            page.evaluate(
                f"(() => {{ const e = document.querySelector('{selector}'); e.value = '{v}'; e.dispatchEvent(new Event('input')); }})()"
            )
            wait(pause)

    page.goto(url, wait_until="networkidle")
    wait(2600)  # hero
    scroll_to("#lab-queue")
    marks["gif_start"] = now()
    click("#q-watch")
    wait(8600)  # one order, animated
    marks["gif_end"] = now()
    slide("#q-informed", list(range(40, -1, -5)), 140)
    wait(900)  # no informed sellers: fills look fine
    slide("#q-informed", list(range(0, 61, 5)), 120)
    wait(900)
    slide("#q-informed", [55, 50, 45, 40], 120)
    slide("#q-ahead", list(range(13600, 999, -900)) + [1000], 70)
    wait(1100)  # short queue
    slide("#q-ahead", list(range(1000, 13601, 900)) + [13600], 60)
    wait(600)
    scroll_to("#lab-fills")
    wait(1200)
    for t in ("MSFT", "AAPL", "INTC"):
        click(f"#f-stock button[data-key={t}]")
        wait(1300)
    scroll_to("#f-rules", 260)
    click("#f-costs button[data-key=r03]")
    wait(1600)
    move_to("#f-rules svg circle")
    wait(1600)
    click("#f-costs button[data-key=gross]")
    wait(800)
    scroll_to("#lab-auction")
    slide("#a-mult", [round(1 + i * 0.05, 2) for i in range(0, 31)], 70)
    wait(1400)
    slide("#a-mult", [round(2.5 - i * 0.05, 2) for i in range(0, 31)], 50)
    wait(800)
    scroll_to("#lab-kelly")
    click("#k-presets button >> nth=0")
    wait(2200)  # Kelly fitted before 2008: wiped out
    click("#k-presets button >> nth=2")
    wait(1800)
    click("#k-presets button >> nth=3")
    wait(2200)  # hindsight Kelly survives
    scroll_to("#lab-game")
    click("#quote")
    wait(1800)
    click("#quote")
    wait(1600)
    scroll_to("#desk")
    wait(2200)
    scroll_to("#about")
    wait(2600)
    page.evaluate("scrollTo({top: 0, behavior: 'smooth'})")
    wait(2200)
    return marks


def card(browser, url: str) -> Path:
    """The link-preview image: the hero alone, at 1200 x 630."""
    page = browser.new_page(viewport={"width": 1200, "height": 630})
    page.goto(url, wait_until="networkidle")
    page.add_style_tag(content="html{zoom:.95} .topnav,main,footer{display:none} .hero{padding-top:44px}")
    page.wait_for_timeout(300)
    out = DOCS / "img" / "markout-card.png"
    page.screenshot(path=str(out))
    page.close()
    return out


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    httpd, url = serve()
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception:  # noqa: BLE001 - fall back to Playwright's Chromium
            browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=SIZE, record_video_dir=tmp, record_video_size=SIZE, color_scheme="light")
        page = ctx.new_page()
        started = time.monotonic()
        page.add_init_script(CURSOR)
        marks = walkthrough(page, url, started)
        video = page.video.path()
        ctx.close()
        print(f"wrote {card(browser, url).relative_to(ROOT)}")
        browser.close()
        webm = Path(tmp) / "demo.webm"
        shutil.copy(video, webm)
        mp4, poster, gif = MEDIA / "markout-demo.mp4", MEDIA / "markout-demo.jpg", MEDIA / "markout-demo.gif"
        run = lambda *args: subprocess.run([ffmpeg, "-y", "-loglevel", "error", *args], check=True)  # noqa: E731
        run(
            "-i",
            str(webm),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(mp4),
        )
        run("-ss", "1.5", "-i", str(mp4), "-frames:v", "1", "-q:v", "3", str(poster))
        start, length = marks["gif_start"] + 0.4, min(marks["gif_end"] - marks["gif_start"], 9.5)
        run(
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{length:.2f}",
            "-i",
            str(mp4),
            "-vf",
            "fps=10,scale=880:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
            str(gif),
        )
    httpd.shutdown()
    for f in (mp4, poster, gif):
        print(f"wrote {f.relative_to(ROOT)} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
