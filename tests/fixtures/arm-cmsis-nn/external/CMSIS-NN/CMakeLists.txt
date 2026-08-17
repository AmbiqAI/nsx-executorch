# NSX-authored stub standing in for ARM-software/CMSIS-NN's build contract:
# a single `cmsis-nn` static library target with a public `Include` dir.
# Not a copy of upstream CMSIS-NN source.

cmake_minimum_required(VERSION 3.15.6)
project(CMSISNN C CXX)

add_library(cmsis-nn STATIC stub.c)
target_include_directories(cmsis-nn PUBLIC "Include")
