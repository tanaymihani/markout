# C++ core of markout: the LOBSTER fill simulator (module G).
#
#   cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release \
#         -Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir) \
#         -DPython_EXECUTABLE=$PWD/.venv/bin/python
#   cmake --build build/cpp -j            (the same as `make cpp`)
#
# The module is written straight into src/markout/lob/, so `from markout.lob import
# _queue_sim` works in the editable install. The install() rule lets the same file
# serve a scikit-build-core wheel build later.
cmake_minimum_required(VERSION 3.18)
project(markout_queue_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_queue_sim bindings.cpp queue_sim.cpp)
target_include_directories(_queue_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Some Command Line Tools installs ship a stub usr/include/c++/v1 next to clang that
# shadows the SDK's complete libc++ headers, so even <cstddef> is "not found". If the
# standard headers are unusable, point clang at the SDK's libc++ instead.
include(CheckIncludeFileCXX)
check_include_file_cxx(cstddef MARKOUT_HAVE_CSTDDEF)
if(NOT MARKOUT_HAVE_CSTDDEF AND APPLE AND EXISTS "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1/cstddef")
  message(STATUS "libc++ headers next to the compiler are incomplete; using ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
  target_compile_options(_queue_sim PRIVATE -nostdinc++ -isystem "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
endif()

# -O3, and nothing that could change floating-point results: no -ffast-math, and no
# fused multiply-add contraction (clang contracts a*b+c by default on arm64). The
# output must be bit-identical to the Python reference.
target_compile_options(_queue_sim PRIVATE -O3 -ffp-contract=off -Wall -Wextra)

# $<1:...> stops multi-config generators from appending a per-config subdirectory.
set_target_properties(_queue_sim PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_SOURCE_DIR}/../src/markout/lob>")

install(TARGETS _queue_sim DESTINATION markout/lob)
