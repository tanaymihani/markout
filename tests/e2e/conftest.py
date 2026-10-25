"""Browser tests (pytest -m e2e). They use the installed Google Chrome when there is
one, else Playwright's own Chromium (`playwright install chromium`)."""

import functools
import http.server
import threading

import pytest

from markout.paths import ROOT

sync_api = pytest.importorskip("playwright.sync_api", reason="needs playwright (pip install -e '.[demo]')")


@pytest.fixture(scope="session")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True)
        except Exception:  # noqa: BLE001 - no system Chrome: use the bundled one
            try:
                b = p.chromium.launch(headless=True)
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"no browser available: {e}")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = ctx.new_page()
    pg.errors = []  # console errors and uncaught exceptions, checked by the tests
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    yield pg
    ctx.close()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        return


class _Server(http.server.ThreadingHTTPServer):
    request_queue_size = 128  # the default backlog of 5 resets some of Chrome's parallel connections


@pytest.fixture(scope="session")
def site_url():
    """docs/ served over HTTP, as GitHub Pages serves it."""
    srv = _Server(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(ROOT / "docs")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    srv.server_close()
