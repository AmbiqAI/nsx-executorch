# NSX-authored test stand-in for AmbiqAI/arm-cmsis-nn.
#
# Mirrors the real package's CMake contract (not its source): idempotently
# create/reuse a `cmsis-nn` target, attach NSX board flags to it, and export
# `nsx::arm_cmsis_nn`. This lets stock ExecuTorch create `cmsis-nn` first
# (via CMSIS_NN_LOCAL_PATH pointed at external/CMSIS-NN below) and this file
# attach to it afterward without ever creating a duplicate target.

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED NSX_BOARD_FLAGS_TARGET)
  message(FATAL_ERROR
    "arm-cmsis-nn fixture: NSX_BOARD_FLAGS_TARGET must be defined before "
    "adding this directory.")
endif()

set(ARM_CMSIS_NN_ROOT "${CMAKE_CURRENT_LIST_DIR}/external/CMSIS-NN"
    CACHE PATH "arm-cmsis-nn fixture: vendored CMSIS-NN stub source")
set(ARM_CMSIS_ROOT "${CMAKE_CURRENT_LIST_DIR}/external/CMSIS_6"
    CACHE PATH "arm-cmsis-nn fixture: vendored CMSIS Core stub source")

if(NOT TARGET cmsis-nn)
  add_subdirectory("${ARM_CMSIS_NN_ROOT}" arm-cmsis-nn-src EXCLUDE_FROM_ALL)
endif()

target_link_libraries(cmsis-nn PUBLIC ${NSX_BOARD_FLAGS_TARGET})
target_include_directories(cmsis-nn PUBLIC
  "${ARM_CMSIS_ROOT}/CMSIS/Core/Include")

add_library(nsx_arm_cmsis_nn INTERFACE)
target_link_libraries(nsx_arm_cmsis_nn INTERFACE cmsis-nn)
add_library(nsx::arm_cmsis_nn ALIAS nsx_arm_cmsis_nn)
