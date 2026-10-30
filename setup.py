"""Builds the C++ fill engine (`markout.lob._queue_sim`) into the wheel.

Everything else lives in pyproject.toml. The extension is optional: without a C++17 compiler the build
skips it, and markout uses the Python engine, which gives the same results more slowly
(`markout.lob.fills.cpp_available()` says which one you have). `make cpp` still builds the same module with
CMake for development.
"""

import os
import subprocess
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Nothing that could change floating-point results: the output must be bit-identical to the Python
# reference, so no -ffast-math and no fused multiply-add contraction (clang contracts by default on arm64).
FLAGS = ["/O2", "/fp:precise"] if sys.platform == "win32" else ["-O3", "-ffp-contract=off"]


def macos_libcxx() -> list[str]:
    """Some Command Line Tools installs ship a stub libc++ next to clang in which even <cstddef> is missing;
    use the SDK's headers then (the same fix as cpp/CMakeLists.txt)."""
    if sys.platform != "darwin":
        return []
    try:
        probe = subprocess.run(
            ["clang++", "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
            input="#include <cstddef>\n",
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return []
        sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return []
    headers = os.path.join(sdk, "usr", "include", "c++", "v1")
    return ["-nostdinc++", "-isystem", headers] if os.path.exists(os.path.join(headers, "cstddef")) else []


engine = Pybind11Extension(
    "markout.lob._queue_sim",
    ["cpp/bindings.cpp", "cpp/queue_sim.cpp"],
    include_dirs=["cpp"],
    cxx_std=17,
    extra_compile_args=FLAGS + macos_libcxx(),
)
engine.optional = True  # a failed compile skips the extension instead of failing the install

setup(ext_modules=[engine], cmdclass={"build_ext": build_ext})
