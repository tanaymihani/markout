PY := .venv/bin/python
PYBIND11_DIR := $(shell $(PY) -m pybind11 --cmakedir 2>/dev/null)

.PHONY: install test test-fast data lobster optiver options cpp bench report all clean

install:
	.venv/bin/pip install -e ".[dev,data,cpp]"

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

# ---- data (never committed; see .gitignore) --------------------------------
data: lobster optiver options

lobster:
	$(PY) scripts/download_lobster.py

optiver:
	$(PY) scripts/download_optiver.py

options:
	$(PY) scripts/snapshot_options.py

# ---- C++ core ---------------------------------------------------------------
cpp:
	cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release \
	      -Dpybind11_DIR=$(PYBIND11_DIR) -DPython_EXECUTABLE=$(abspath $(PY))
	cmake --build build/cpp -j

bench: cpp
	$(PY) -m markout.lob.bench

# ---- reports: every number in reports/ comes from these commands ------------
report:
	$(PY) -m markout.auction.report
	$(PY) -m markout.lob.report
	$(PY) -m markout.games.report
	$(PY) -m markout.options.report
	$(PY) -m markout.decision.contest

all: data cpp test report

clean:
	rm -rf build .pytest_cache .hypothesis
