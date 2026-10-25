PY := .venv/bin/python
PYBIND11_DIR := $(shell $(PY) -m pybind11 --cmakedir 2>/dev/null)

.PHONY: install install-locked test test-fast cov e2e lint format typecheck check lock data lobster optiver \
        options vol cpp bench report readme site api-docs demo demo-video desk public-desk docker all clean

install:
	.venv/bin/pip install -e ".[dev,data,cpp]"

install-locked:  # the exact versions CI uses
	uv pip install --python $(PY) -r requirements.lock
	uv pip install --python $(PY) --no-deps -e .

# ---- checks (CI runs lint, typecheck, cov and e2e) ---------------------------
test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

cov:
	$(PY) -m pytest --cov --cov-report=term --cov-report=html

e2e:  # browser tests; needs the demo extra and Chrome or `playwright install chromium`
	$(PY) -m pytest -m e2e

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

typecheck:
	.venv/bin/mypy

check: lint typecheck test e2e

api-docs:  # docs/api/, the API reference on the site
	$(PY) scripts/api_docs.py

# Re-resolve requirements.lock and the image's pins. Existing pins are kept unless they
# must change; add --upgrade to the commands to move everything, then run `make check`.
lock:
	uv pip compile pyproject.toml --extra dev --extra cpp --extra demo --universal --python-version 3.11 \
	    -o requirements.lock --custom-compile-command "make lock"
	uv pip compile deploy/requirements-desk.in --python-version 3.11 --python-platform linux --generate-hashes \
	    -o deploy/requirements-desk.txt --custom-compile-command "make lock"

# ---- data (never committed; see .gitignore) --------------------------------
data: lobster optiver options vol

lobster:
	$(PY) scripts/download_lobster.py

optiver:
	$(PY) scripts/download_optiver.py

options:
	$(PY) scripts/snapshot_options.py

vol:
	$(PY) -m markout.vol.data

# ---- C++ core ---------------------------------------------------------------
cpp:
	cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release \
	      -Dpybind11_DIR=$(PYBIND11_DIR) -DPython_EXECUTABLE=$(abspath $(PY))
	cmake --build build/cpp -j

bench: cpp
	$(PY) -m markout.lob.bench

# ---- reports: every number in reports/ comes from these commands ------------
report:
	@if [ -f data/processed/optiver/train.parquet ]; then $(PY) -m markout.auction.report; \
	  else echo "skipping report 01: run 'make optiver' first (needs a Kaggle login)"; fi
	$(PY) -m markout.lob.report
	$(PY) -m markout.games.report
	$(PY) -m markout.options.report
	@if [ -f data/raw/vol/spx.parquet ]; then $(PY) -m markout.vol.report; \
	  else echo "skipping report 06: run 'make vol' first"; fi
	$(PY) -m markout.decision.contest
	$(PY) -m markout.cup demo
	$(PY) -m markout.site
	$(PY) -m markout.readme

readme:
	$(PY) -m markout.readme

site:
	$(PY) -m markout.site

demo:
	$(PY) -m markout.cup demo

demo-video:
	$(PY) scripts/record_demo.py

desk:
	$(PY) -m markout.cup serve --paper

public-desk:  # the hosted desk, one sandbox per visitor: http://127.0.0.1:7860
	$(PY) -m markout.cup public

docker:
	docker build -t markout-desk .
	@echo "run it: docker run --rm -p 7860:7860 markout-desk"

all: data cpp test report

clean:
	rm -rf build .pytest_cache .hypothesis .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
