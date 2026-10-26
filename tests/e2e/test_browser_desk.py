"""The desk on the published site, running in Chrome on Pyodide (downloaded from its CDN)."""

import pytest

pytestmark = pytest.mark.e2e
expect = pytest.importorskip("playwright.sync_api").expect


def test_the_desk_runs_in_the_browser(page, site_url):
    page.goto(site_url + "desk/")
    page.wait_for_selector("button[data-act=approve]:not([disabled])", timeout=180_000)  # Pyodide, numpy, pandas
    expect(page.locator("#sbx")).to_contain_text("seed")
    page.locator("button[data-act=approve]:not([disabled])").first.click()
    expect(page.locator("#positions table")).to_be_visible()
    page.click("button[data-time='24']")
    expect(page.locator("#meta")).to_contain_text("2026-10-02")
    page.on("dialog", lambda d: d.accept())  # "Start over with a fresh sandbox?"
    page.click("button[data-time='reset']")
    expect(page.locator("#meta")).to_contain_text("2026-10-01")
    expect(page.locator("#positions")).to_have_text("None")
    assert page.errors == []
