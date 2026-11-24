cmake_minimum_required(VERSION 3.24)
project(nsx_executorch_real_provider_smoke LANGUAGES C CXX)

# nsx configure always passes -DCMAKE_BUILD_TYPE=Release (neuralspotx
# project_config.py). Default the harness the same way so it exercises the
# Release-only flag edits stock ExecuTorch makes; an empty build type used to
# leave the whole subtree at -O0 while EXECUTORCH_OPTIMIZE_SIZE claimed -Os.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

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
# The production interface also carries the app's non-ABI policy flags
# (nsx_toolchain_flags.cmake): -g -O3 -ffast-math and section/codegen flags.
# Include them so the module's ABI-vs-policy filtering is exercised by the
# harness exactly as it is in a bootstrapped NSX app.
target_compile_options(nsx_board_flags INTERFACE
  -fshort-enums
  -ffunction-sections
  -fdata-sections
  -fomit-frame-pointer
  -fno-exceptions
  -Wall
  -g
  -O3
  -ffast-math)
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

# A firmware that links nsx_executorch also links every kernel its registration
# libraries name; a selection that names an operator whose kernel is not built
# fails only then, at the app's link. nsx_executorch itself needs the Ambiq HAL,
# which this harness does not provide, so link its registration libraries (whole,
# as their static initializers are) with the ExecuTorch runtime into a bare
# executable instead. Building it also runs the exact kernel count.
option(NSX_EXECUTORCH_TEST_LINK_CHECK
  "Link nsx_executorch's kernel registration libraries into a bare executable" OFF)
if(NSX_EXECUTORCH_TEST_LINK_CHECK)
  get_property(_nsx_executorch_test_registrations GLOBAL PROPERTY NSX_EXECUTORCH_REGISTRATION_LIBS)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/link_check.c" "int main(void) { return 0; }\n")
  add_executable(nsx_executorch_link_check "${CMAKE_CURRENT_BINARY_DIR}/link_check.c")
  target_link_libraries(nsx_executorch_link_check PRIVATE nsx::board_flags executorch)
  if(_nsx_executorch_test_registrations)
    # Whole, by hand: CMake's WHOLE_ARCHIVE link feature is not defined for a
    # Generic (bare-metal) system. Each library is also linked by name, for the
    # kernel libraries it depends on.
    set(_nsx_executorch_test_whole "")
    foreach(_lib IN LISTS _nsx_executorch_test_registrations)
      list(APPEND _nsx_executorch_test_whole "$<TARGET_FILE:${_lib}>")
    endforeach()
    target_link_libraries(nsx_executorch_link_check PRIVATE
      -Wl,--whole-archive ${_nsx_executorch_test_whole} -Wl,--no-whole-archive
      ${_nsx_executorch_test_registrations})
  endif()
  # The board flags are compile options only; the link needs the same ABI to pick the matching libc.
  target_link_options(nsx_executorch_link_check PRIVATE
    -mthumb "-mcpu=${NSX_EXECUTORCH_TEST_CPU}"
    $<$<STREQUAL:${NSX_EXECUTORCH_TEST_CPU},cortex-m55>:-mfloat-abi=hard>
    --specs=nosys.specs -Wl,--gc-sections)
  add_dependencies(nsx_executorch_link_check nsx_executorch_kernel_count)
endif()
