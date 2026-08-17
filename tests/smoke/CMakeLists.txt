cmake_minimum_required(VERSION 3.24)
project(nsx_executorch_configure_smoke LANGUAGES C CXX)

add_library(nsx_core INTERFACE)
add_library(nsx::core ALIAS nsx_core)
add_library(nsx_cmsis_core INTERFACE)
add_library(nsx::cmsis_core ALIAS nsx_cmsis_core)
add_library(nsx_soc_hal INTERFACE)
add_library(nsx::soc_hal ALIAS nsx_soc_hal)
add_library(nsx_board_flags INTERFACE)
add_library(nsx::board_flags ALIAS nsx_board_flags)
# A real NSX app always installs its board-flags target as part of its own
# export set (see neuralspotx boards/*/board.cmake, which sets an
# EXPORT_NAME on this target for that purpose). Mirror that here: stock
# ExecuTorch's own install(EXPORT ExecuTorchTargets ...) call requires every
# target that cmsis-nn PUBLIC-links (including this board-flags target, via
# the arm provider) to belong to *some* export set, not necessarily
# ExecuTorchTargets itself.
include(GNUInstallDirs)
install(TARGETS nsx_board_flags EXPORT nsx_executorch_smoke_test_targets)
install(EXPORT nsx_executorch_smoke_test_targets
        FILE NsxExecutorchSmokeTestTargets.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NsxExecutorchSmokeTest")

set(NSX_EXECUTORCH_CMSIS_NN_PROVIDER arm CACHE STRING "" FORCE)
set(NSX_EXECUTORCH_ENABLE_PROFILING ON CACHE BOOL "" FORCE)
get_filename_component(_arm_cmsis_nn_fixture_root
  "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/arm-cmsis-nn" ABSOLUTE)
set(NSX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(NSX_APP_MODULE_DIR_arm_cmsis_nn "fixtures/arm-cmsis-nn")
set(NSX_BOARD_FLAGS_TARGET nsx::board_flags)
add_subdirectory("${_arm_cmsis_nn_fixture_root}" arm-cmsis-nn-provider)
add_subdirectory("../.." nsx-executorch)

if(NOT TARGET nsx::executorch)
  message(FATAL_ERROR "nsx::executorch target was not created")
endif()
if(NOT EXECUTORCH_ENABLE_EVENT_TRACER)
  message(FATAL_ERROR "profiling did not enable stock ExecuTorch EventTracer")
endif()
if(NOT CMSIS_NN_LOCAL_PATH MATCHES "cmsis-nn-provider$")
  message(FATAL_ERROR "arm-provider adapter was not selected")
endif()
if(NOT EXISTS "${CMSIS_NN_LOCAL_PATH}/Include/arm_nnfunctions.h")
  message(FATAL_ERROR "arm provider headers were not materialized")
endif()
if(NOT TARGET nsx::arm_cmsis_nn)
  message(FATAL_ERROR "arm provider did not export nsx::arm_cmsis_nn")
endif()
if(NOT TARGET cmsis-nn)
  message(FATAL_ERROR "stock ExecuTorch did not create the cmsis-nn target")
endif()
if(NOT FETCHCONTENT_FULLY_DISCONNECTED)
  message(FATAL_ERROR "CMake dependency fetching was not disabled")
endif()
