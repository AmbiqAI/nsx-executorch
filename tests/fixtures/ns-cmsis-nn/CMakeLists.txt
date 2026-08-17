# NSX-authored test stand-in for AmbiqAI/ns-cmsis-nn's root package.

cmake_minimum_required(VERSION 3.24)
project(CMSISNN C CXX)

add_library(cmsis-nn STATIC stub.c)
target_include_directories(cmsis-nn PUBLIC "Include")
