PY := .venv/bin/python
PYBIND11_DIR := $(shell $(PY) -m pybind11 --cmakedir 2>/dev/null)

.PHONY: install test test-fast data lobster optiver options vol cpp bench report readme demo desk all clean

install:
	.venv/bin/pip install -e ".[dev,data,cpp]"

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

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
	$(PY) -m markout.readme

readme:
	$(PY) -m markout.readme

demo:
	$(PY) -m markout.cup demo

desk:
	$(PY) -m markout.cup serve --paper

all: data cpp test report

clean:
	rm -rf build .pytest_cache .hypothesis
