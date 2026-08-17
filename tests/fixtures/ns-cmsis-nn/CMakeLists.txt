# NSX-authored test stand-in for AmbiqAI/ns-cmsis-nn's ROOT package (its
# separate nsx/ integration entry point exports a different target and is
# intentionally not mirrored here; see tests/fixtures/README.md). Mirrors
# the drop-in contract of directly creating the `cmsis-nn` target. Not a
# copy of any Ambiq source.

cmake_minimum_required(VERSION 3.24)
project(CMSISNN C CXX)

add_library(cmsis-nn STATIC stub.c)
target_include_directories(cmsis-nn PUBLIC "Include")
