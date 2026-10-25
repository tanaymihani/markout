"""The published site is exactly what the generator builds from the committed results."""

import re

from markout import site
from markout.paths import ROOT

BUILT = re.compile(r"Built [A-Z][a-z]+ \d{4}\.")  # the build month, the one thing allowed to differ


def test_site_is_up_to_date():
    fresh = BUILT.sub("Built <month>.", site.render())
    page = BUILT.sub("Built <month>.", (ROOT / "docs" / "index.html").read_text())
    assert fresh == page, "docs/index.html is out of date: run `markout site`"
