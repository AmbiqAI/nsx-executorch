cmake_minimum_required(VERSION 3.24)
project(nsx_executorch_standalone_smoke LANGUAGES C CXX)

if(NOT NSX_EXECUTORCH_TEST_PROVIDER MATCHES "^(arm|ns)$")
  message(FATAL_ERROR
    "Set NSX_EXECUTORCH_TEST_PROVIDER to arm or ns.")
endif()

add_library(nsx_core INTERFACE)
add_library(nsx::core ALIAS nsx_core)
add_library(nsx_cmsis_core INTERFACE)
add_library(nsx::cmsis_core ALIAS nsx_cmsis_core)
add_library(nsx_soc_hal INTERFACE)
add_library(nsx::soc_hal ALIAS nsx_soc_hal)
add_library(nsx_board_flags INTERFACE)
add_library(nsx::board_flags ALIAS nsx_board_flags)
include(GNUInstallDirs)
install(TARGETS nsx_board_flags EXPORT nsx_executorch_standalone_targets)
install(EXPORT nsx_executorch_standalone_targets
        FILE NsxExecutorchStandaloneTargets.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NsxExecutorchStandalone")

set(NSX_EXECUTORCH_CMSIS_NN_PROVIDER "${NSX_EXECUTORCH_TEST_PROVIDER}"
    CACHE STRING "" FORCE)
get_filename_component(_source_root "../.." ABSOLUTE
  BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(NSX_EXECUTORCH_SOURCE_ROOT "${_source_root}/external/executorch"
    CACHE PATH "" FORCE)

if(NSX_EXECUTORCH_TEST_PROVIDER STREQUAL "arm")
  get_filename_component(_provider_root
    "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/arm-cmsis-nn" ABSOLUTE)
  set(NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT "${_provider_root}"
      CACHE PATH "" FORCE)
else()
  get_filename_component(_provider_root
    "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/ns-cmsis-nn/nsx" ABSOLUTE)
  set(NSX_EXECUTORCH_NS_CMSIS_NN_ROOT "${_provider_root}"
      CACHE PATH "" FORCE)
endif()

add_subdirectory("../.." nsx-executorch)

if(NOT TARGET nsx::executorch OR NOT TARGET cmsis-nn)
  message(FATAL_ERROR "standalone provider integration failed")
endif()
if(DEFINED NSX_ROOT)
  message(FATAL_ERROR "standalone smoke must not define NSX_ROOT")
endif()
