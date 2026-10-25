"""Build the API reference (docs/api/) with pdoc: the library, not the report builders.

    .venv/bin/python scripts/api_docs.py        # or: make api-docs

pdoc prints module constants with their values, and some of them are paths, so the
repository's absolute location is rewritten to "." and the pages read the same wherever
the repository was cloned.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from markout import __version__
from markout.paths import ROOT

OUT = ROOT / "docs" / "api"
# markout.evaluation's __all__ lists functions, which hides its submodules; name them so
# their module docs (the method and the references) get pages too
EXTRA = tuple(f"markout.evaluation.{m}" for m in ("dsr", "bootstrap", "registry", "holdout", "forecast_tests"))
SKIP = (
    "markout.lob._queue_sim",
    "markout.*.report",
    "markout.lob.figures",
    "markout.lob.bench",
    "markout.site",
    "markout.readme",
    "markout.cup.demo",
    "markout.cup.dryrun",
    "markout.__main__",
    "markout.cup.__main__",
)


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pdoc",
            "markout",
            *EXTRA,
            *(f"!{m}" for m in SKIP),
            "--no-show-source",
            "--no-search",
            "--docformat",
            "markdown",
            "--edit-url",
            "markout=https://github.com/tanaymihani/markout/blob/main/src/markout/",
            "--footer-text",
            f"Markout {__version__} · MIT licence",
            "-o",
            str(OUT),
        ],
        check=True,
    )
    for page in OUT.rglob("*.html"):
        text = page.read_text()
        if str(ROOT) in text:
            page.write_text(text.replace(str(ROOT), "."))
    print(f"wrote {OUT.relative_to(ROOT)}/ ({sum(1 for _ in OUT.rglob('*.html'))} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
