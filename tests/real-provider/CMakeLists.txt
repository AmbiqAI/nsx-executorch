cmake_minimum_required(VERSION 3.24)
project(nsx_executorch_real_provider_smoke LANGUAGES C CXX)

foreach(_required
    NSX_EXECUTORCH_TEST_ROOT
    NSX_EXECUTORCH_TEST_PROVIDER
    NSX_EXECUTORCH_TEST_ARM_ROOT)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

add_library(nsx_core INTERFACE)
add_library(nsx::core ALIAS nsx_core)
add_library(nsx_cmsis_core INTERFACE)
add_library(nsx::cmsis_core ALIAS nsx_cmsis_core)
add_library(nsx_soc_hal INTERFACE)
add_library(nsx::soc_hal ALIAS nsx_soc_hal)
add_library(nsx_board_flags INTERFACE)
add_library(nsx::board_flags ALIAS nsx_board_flags)
set(NSX_EXECUTORCH_TEST_CPU "cortex-m55" CACHE STRING
    "CPU used for real-provider compilation")
# Mirror nsx_apply_toolchain_flags() in neuralspotx: for Cortex-M55 the app
# passes no -mfpu (NSX_FPU is Cortex-M4 only). An explicit -mfpu=fpv5-sp-d16
# here would downgrade clang's __ARM_FEATURE_MVE from 3 to 1 (integer-only),
# breaking ExecuTorch's Helium float paths under ATfE; GCC ignores -mfpu when
# deriving MVE, which is why the GCC leg never noticed.
if(NSX_EXECUTORCH_TEST_CPU STREQUAL "cortex-m55")
  target_compile_options(nsx_board_flags INTERFACE
    -mthumb
    -mcpu=cortex-m55
    -mfloat-abi=hard)
else()
  target_compile_options(nsx_board_flags INTERFACE
    "-mcpu=${NSX_EXECUTORCH_TEST_CPU}"
    -mthumb)
endif()
include(GNUInstallDirs)
install(TARGETS nsx_board_flags EXPORT nsx_executorch_real_provider_targets)
install(EXPORT nsx_executorch_real_provider_targets
        FILE NsxExecutorchRealProviderTargets.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NsxExecutorchRealProvider")

set(NSX_BOARD_FLAGS_TARGET nsx::board_flags)
set(NSX_EXECUTORCH_CMSIS_NN_PROVIDER "${NSX_EXECUTORCH_TEST_PROVIDER}"
    CACHE STRING "" FORCE)
set(NSX_EXECUTORCH_ENABLE_PROFILING ON CACHE BOOL "" FORCE)
set(NSX_EXECUTORCH_TEST_MODE "standalone" CACHE STRING
    "Provider integration mode: standalone or nsx")
target_include_directories(nsx_cmsis_core INTERFACE
  "${NSX_EXECUTORCH_TEST_ARM_ROOT}/external/CMSIS_6/CMSIS/Core/Include")

if(NSX_EXECUTORCH_TEST_PROVIDER STREQUAL "arm")
  set(NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT
      "${NSX_EXECUTORCH_TEST_ARM_ROOT}" CACHE PATH "" FORCE)
  if(NSX_EXECUTORCH_TEST_MODE STREQUAL "nsx")
    set(NSX_ROOT "${CMAKE_CURRENT_BINARY_DIR}/nsx-app")
    add_subdirectory(
      "${NSX_EXECUTORCH_TEST_ARM_ROOT}"
      arm-cmsis-nn-provider)
  endif()
elseif(NSX_EXECUTORCH_TEST_PROVIDER STREQUAL "ns")
  if(NOT DEFINED NSX_EXECUTORCH_TEST_NS_ROOT)
    message(FATAL_ERROR "NSX_EXECUTORCH_TEST_NS_ROOT is required")
  endif()
  set(NSX_EXECUTORCH_NS_CMSIS_NN_ROOT
      "${NSX_EXECUTORCH_TEST_NS_ROOT}" CACHE PATH "" FORCE)
  set(CMSIS_PATH
      "${NSX_EXECUTORCH_TEST_ARM_ROOT}/external/CMSIS_6"
      CACHE PATH "" FORCE)
  if(NSX_EXECUTORCH_TEST_MODE STREQUAL "nsx")
    set(NSX_ROOT "${CMAKE_CURRENT_BINARY_DIR}/nsx-app")
    add_subdirectory(
      "${NSX_EXECUTORCH_TEST_NS_ROOT}/nsx"
      ns-cmsis-nn-provider)
  endif()
else()
  message(FATAL_ERROR "NSX_EXECUTORCH_TEST_PROVIDER must be arm or ns")
endif()

add_subdirectory("${NSX_EXECUTORCH_TEST_ROOT}" nsx-executorch)
