"""The published site is exactly what the generator builds from the committed results."""

import io
import re
import zipfile

from markout import site
from markout.paths import ROOT

BUILT = re.compile(r"Built [A-Z][a-z]+ \d{4}\.")  # the build month, the one thing allowed to differ


def test_site_is_up_to_date():
    fresh = BUILT.sub("Built <month>.", site.render())
    page = BUILT.sub("Built <month>.", (ROOT / "docs" / "index.html").read_text())
    assert fresh == page, "docs/index.html is out of date: run `markout site`"


def test_desk_page_is_up_to_date():
    assert site.desk_page() == (ROOT / "docs" / "desk" / "index.html").read_text(), "run `markout site`"


def test_desk_bundle_is_up_to_date():
    def contents(data: bytes) -> dict[str, bytes]:  # compare files, not zip bytes (zlib differs by platform)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return {n: z.read(n) for n in z.namelist()}

    committed = (ROOT / "docs" / "desk" / "markout-desk.zip").read_bytes()
    assert contents(site.desk_bundle()) == contents(committed), (
        "docs/desk/markout-desk.zip is out of date: run `markout site`"
    )
