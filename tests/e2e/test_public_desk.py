"""The hosted desk in a real browser: a sandbox per tab, trades, time, and a strict CSP."""

import threading

import pytest

from markout.cup.public import PublicDesk, Sessions

expect = pytest.importorskip("playwright.sync_api").expect

pytestmark = pytest.mark.e2e


@pytest.fixture
def desk_url(tmp_path):
    sessions = Sessions(tmp_path)
    srv = PublicDesk(sessions, log=lambda e: None).serve("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    srv.server_close()
    sessions.close()


def test_a_visitor_trades_moves_the_clock_and_keeps_the_sandbox(page, desk_url):
    page.goto(desk_url)
    page.wait_for_selector("button[data-act=approve]:not([disabled])")
    assert "seed" in page.inner_text("#sbx")
    page.locator("button[data-act=approve]:not([disabled])").first.click()
    page.wait_for_selector("#positions table")
    page.click("button[data-time='24']")
    # (the page's CSP forbids eval, so waits use locators rather than wait_for_function)
    expect(page.locator("#meta")).to_contain_text("2026-10-02")
    page.reload()  # same tab, same sandbox
    page.wait_for_selector("#positions table")
    expect(page.locator("#meta")).to_contain_text("2026-10-02")
    assert page.errors == []  # includes CSP violations, which Chrome reports as console errors


def test_two_tabs_get_two_sandboxes(browser, desk_url):
    ctx = browser.new_context()
    a, b = ctx.new_page(), ctx.new_page()
    for pg in (a, b):
        pg.goto(desk_url)
        pg.wait_for_selector("button[data-act=approve]:not([disabled])")
    a.locator("button[data-act=approve]:not([disabled])").first.click()
    a.wait_for_selector("#positions table")
    b.reload()
    expect(b.locator("#positions")).to_have_text("None")
    ctx.close()
