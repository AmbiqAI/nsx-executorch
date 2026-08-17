cmake_minimum_required(VERSION 3.19)

add_library(nsx_cmsis_nn STATIC ../stub.c)
add_library(nsx::cmsis_nn ALIAS nsx_cmsis_nn)
target_include_directories(nsx_cmsis_nn PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}/../Include")
