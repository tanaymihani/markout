"""The published site loads cleanly and every lab responds to its controls."""

import pytest

pytestmark = pytest.mark.e2e


def test_site_loads_without_errors_and_the_labs_respond(page, site_url):
    page.goto(site_url, wait_until="domcontentloaded")
    page.wait_for_selector("#q-results:not([hidden])")
    assert page.locator("svg").count() >= 4  # the labs' charts are drawn
    assert page.locator(".hero a[href='desk/']").count() == 1 and page.locator("#desk a[href='desk/']").count() >= 1

    before = page.inner_text("#q-fifo-mk")
    page.fill("#q-ahead", "2000")  # a short queue: fills are less adversely selected
    page.dispatch_event("#q-ahead", "input")
    page.wait_for_function(f"document.querySelector('#q-fifo-mk').textContent !== {before!r}")

    first = page.inner_text("#f-summary")
    page.locator("#f-stock button").last.click()
    assert page.inner_text("#f-summary") != first

    sharpe = page.inner_text("#a-sharpe")
    page.fill("#a-mult", "3")
    page.dispatch_event("#a-mult", "input")
    assert page.inner_text("#a-sharpe") != sharpe

    page.locator("#k-presets button").first.click()  # full Kelly on the pre-2008 data
    assert page.inner_text("#k-final") == "wiped out"
    assert "2008" in page.inner_text("#k-ruin")
    page.locator("#k-presets button").nth(3).click()  # the hindsight optimum survives
    assert page.inner_text("#k-final").startswith("×") and page.inner_text("#k-ruin") == "survived"

    assert page.errors == []


def test_site_works_at_phone_width(browser, site_url):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)
    pg = ctx.new_page()
    pg.goto(site_url, wait_until="domcontentloaded")
    pg.wait_for_selector("#q-results:not([hidden])")
    overflow = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    ctx.close()
    assert overflow <= 1  # no horizontal page scroll
