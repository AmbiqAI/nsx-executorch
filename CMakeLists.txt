cmake_minimum_required(VERSION 3.24)

if(TARGET nsx::executorch)
  return()
endif()

foreach(_required_target nsx::core nsx::cmsis_core nsx::soc_hal nsx::board_flags)
  if(NOT TARGET ${_required_target})
    message(FATAL_ERROR
      "nsx-executorch: required target '${_required_target}' is not defined.")
  endif()
endforeach()

get_filename_component(NSX_EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
set(NSX_EXECUTORCH_SOURCE_ROOT
    "${NSX_EXECUTORCH_ROOT}/external/executorch" CACHE PATH
    "Path to an unmodified ExecuTorch source checkout")
if(NOT EXISTS "${NSX_EXECUTORCH_SOURCE_ROOT}/version.txt")
  message(FATAL_ERROR
    "nsx-executorch: stock ExecuTorch source is missing. Initialize the "
    "external/executorch submodule or set NSX_EXECUTORCH_SOURCE_ROOT.")
endif()

set(NSX_EXECUTORCH_CMSIS_NN_PROVIDER "arm" CACHE STRING
    "CMSIS-NN provider consumed by stock ExecuTorch: arm or ns")
set_property(CACHE NSX_EXECUTORCH_CMSIS_NN_PROVIDER PROPERTY STRINGS arm ns)
set(NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT "" CACHE PATH
    "Path to a resolved arm-cmsis-nn NSX module checkout (provider=arm)")
set(NSX_EXECUTORCH_NS_CMSIS_NN_ROOT "" CACHE PATH
    "Path to a resolved nsx-cmsis-nn module directory or repository root")
set(NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST "" CACHE STRING
    "Comma-separated portable ATen fallback operators; empty disables them")
set(NSX_EXECUTORCH_ENABLE_PROFILING OFF CACHE BOOL
    "Enable EventTracer-backed per-operator callbacks")
set(NSX_EXECUTORCH_ENABLE_NS_OPS OFF CACHE BOOL
    "Build and register the cortex_m_ns:: operators backed by ns-cmsis-nn additional kernels (requires provider=ns)")
set(NSX_EXECUTORCH_OPTIMIZATION "speed" CACHE STRING
    "Optimization objective for ExecuTorch code: speed (-O3) or size (-Os)")
set_property(CACHE NSX_EXECUTORCH_OPTIMIZATION PROPERTY STRINGS speed size)
set(NSX_EXECUTORCH_ENABLE_PROGRAM_VERIFICATION OFF CACHE BOOL
    "Build and run the flatbuffer program verifier at Program::load")

if(NSX_EXECUTORCH_OPTIMIZATION STREQUAL "speed")
  set(_nsx_executorch_opt_flag -O3)
elseif(NSX_EXECUTORCH_OPTIMIZATION STREQUAL "size")
  set(_nsx_executorch_opt_flag -Os)
else()
  message(FATAL_ERROR
    "nsx-executorch: NSX_EXECUTORCH_OPTIMIZATION must be speed or size.")
endif()

if(NSX_EXECUTORCH_ENABLE_NS_OPS
   AND NOT NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns")
  message(FATAL_ERROR
    "nsx-executorch: NSX_EXECUTORCH_ENABLE_NS_OPS requires "
    "NSX_EXECUTORCH_CMSIS_NN_PROVIDER=ns; the cortex_m_ns:: kernels only "
    "exist in ns-cmsis-nn.")
endif()

# Resolve a required NSX module's checkout directory the same way the real
# NSX tooling does (see neuralspotx's nsx_app_bootstrap.cmake,
# nsx_module_dir_for_name()): an explicit override wins; otherwise, when this
# package is consumed from within a bootstrapped NSX app, NSX_ROOT is a
# plain (non-cache) variable inherited from that app's top-level directory
# scope, and NSX_APP_MODULE_DIR_<module> overrides a mismatched project/module
# name (e.g. module nsx-cmsis-nn is vendored from project ns-cmsis-nn). This
# never guesses sibling directories; standalone/test consumers must supply
# the explicit override.
function(_nsx_executorch_resolve_module_root out_var module_name override_value)
  if(override_value)
    set(${out_var} "${override_value}" PARENT_SCOPE)
    return()
  endif()
  if(DEFINED NSX_ROOT AND NSX_ROOT)
    string(REPLACE "-" "_" _module_var "${module_name}")
    set(_dir_var "NSX_APP_MODULE_DIR_${_module_var}")
    if(DEFINED ${_dir_var})
      set(${out_var} "${NSX_ROOT}/${${_dir_var}}" PARENT_SCOPE)
    else()
      set(${out_var} "${NSX_ROOT}/modules/${module_name}" PARENT_SCOPE)
    endif()
    return()
  endif()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

function(_nsx_executorch_normalize_ns_root out_var module_dir)
  get_filename_component(_module_dir "${module_dir}" ABSOLUTE)
  if(EXISTS "${_module_dir}/Include/arm_nnfunctions.h"
     AND EXISTS "${_module_dir}/CMakeLists.txt")
    set(${out_var} "${_module_dir}" PARENT_SCOPE)
  elseif(EXISTS "${_module_dir}/../Include/arm_nnfunctions.h"
         AND EXISTS "${_module_dir}/../CMakeLists.txt")
    get_filename_component(_source_root "${_module_dir}/.." ABSOLUTE)
    set(${out_var} "${_source_root}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

if(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "arm")
  _nsx_executorch_resolve_module_root(_nsx_executorch_arm_cmsis_nn_root
      "arm-cmsis-nn" "${NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT}")
  if(NOT EXISTS "${_nsx_executorch_arm_cmsis_nn_root}/CMakeLists.txt")
    message(FATAL_ERROR
      "nsx-executorch: provider=arm requires the arm-cmsis-nn NSX module. "
      "Resolve it as an NSX module dependency or set "
      "NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT to a checkout of "
      "AmbiqAI/arm-cmsis-nn.")
  endif()
  # arm-cmsis-nn vendors upstream CMSIS-NN and CMSIS_6 under its own
  # external/. Point stock ExecuTorch straight at that pinned source so it
  # remains the single place that creates the `cmsis-nn` target.
  set(_nsx_executorch_cmsis_nn_root
      "${_nsx_executorch_arm_cmsis_nn_root}/external/CMSIS-NN")
  set(_nsx_executorch_cmsis_root
      "${_nsx_executorch_arm_cmsis_nn_root}/external/CMSIS_6")
elseif(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns")
  _nsx_executorch_resolve_module_root(_nsx_executorch_ns_cmsis_nn_module_dir
      "nsx-cmsis-nn" "${NSX_EXECUTORCH_NS_CMSIS_NN_ROOT}")
  _nsx_executorch_normalize_ns_root(_nsx_executorch_ns_cmsis_nn_root
      "${_nsx_executorch_ns_cmsis_nn_module_dir}")
  if(NOT _nsx_executorch_ns_cmsis_nn_root)
    message(FATAL_ERROR
      "nsx-executorch: provider=ns requires the nsx-cmsis-nn NSX module "
      "(project ns-cmsis-nn). Resolve it as an NSX module dependency or set "
      "NSX_EXECUTORCH_NS_CMSIS_NN_ROOT to either the repository root or its "
      "nsx/ module directory.")
  endif()
  set(_nsx_executorch_cmsis_nn_root
      "${_nsx_executorch_ns_cmsis_nn_root}")
  # A drop-in ns-cmsis-nn package owns its CMSIS Core discovery contract.
  set(_nsx_executorch_cmsis_root "")
else()
  message(FATAL_ERROR
    "nsx-executorch: NSX_EXECUTORCH_CMSIS_NN_PROVIDER must be arm or ns.")
endif()

if(NOT EXISTS "${_nsx_executorch_cmsis_nn_root}/CMakeLists.txt")
  message(FATAL_ERROR
    "nsx-executorch: CMSIS-NN source is missing at "
    "'${_nsx_executorch_cmsis_nn_root}'.")
endif()

# In a bootstrapped NSX app the selected provider must precede this module in
# nsx.yml. That lets NSX configure the provider exactly once before stock
# ExecuTorch consumes it. Standalone consumers can instead supply a root
# override; the provider adapter below configures that source directly.
if(DEFINED NSX_ROOT AND NSX_ROOT)
  if(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "arm"
     AND NOT TARGET nsx::arm_cmsis_nn)
    message(FATAL_ERROR
      "nsx-executorch: list arm-cmsis-nn before nsx-executorch in nsx.yml.")
  elseif(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns"
         AND NOT TARGET nsx::cmsis_nn)
    message(FATAL_ERROR
      "nsx-executorch: list nsx-cmsis-nn before nsx-executorch in nsx.yml.")
  elseif(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns"
         AND TARGET cmsis-nn)
    message(FATAL_ERROR
      "nsx-executorch: provider=ns found an existing cmsis-nn target. "
      "Do not include arm-cmsis-nn or another CMSIS-NN provider.")
  endif()
endif()
if(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "arm"
   AND TARGET cmsis-nn
   AND NOT TARGET nsx::arm_cmsis_nn)
  message(FATAL_ERROR
    "nsx-executorch: provider=arm found an unrecognized cmsis-nn target.")
elseif(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns"
       AND TARGET cmsis-nn
       AND TARGET nsx::arm_cmsis_nn)
  message(FATAL_ERROR
    "nsx-executorch: provider=ns cannot reuse arm-cmsis-nn.")
endif()

# ExecuTorch invokes torchgen during configure and build. Wrap the selected
# host interpreter so both phases see the pinned, vendored host tooling without
# modifying ExecuTorch or requiring a full PyTorch installation.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(NSX_EXECUTORCH_HOST_PYTHON "${Python3_EXECUTABLE}")
set(NSX_EXECUTORCH_TORCHGEN_ROOT "${NSX_EXECUTORCH_ROOT}/tools/python")
set(_nsx_executorch_python
    "${CMAKE_CURRENT_BINARY_DIR}/nsx-executorch-python")
configure_file(
  "${NSX_EXECUTORCH_ROOT}/tools/python_with_torchgen.sh.in"
  "${_nsx_executorch_python}" @ONLY)
file(CHMOD "${_nsx_executorch_python}"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
set(PYTHON_EXECUTABLE "${_nsx_executorch_python}" CACHE FILEPATH
    "Host Python with nsx-executorch torchgen path" FORCE)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import torchgen"
  RESULT_VARIABLE _nsx_executorch_torchgen_result
  ERROR_VARIABLE _nsx_executorch_torchgen_error)
if(NOT _nsx_executorch_torchgen_result EQUAL 0)
  message(FATAL_ERROR
    "nsx-executorch: pinned torchgen could not be imported: "
    "${_nsx_executorch_torchgen_error}")
endif()

if(NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST)
  set(_nsx_executorch_build_portable_ops ON)
else()
  set(_nsx_executorch_build_portable_ops OFF)
endif()

# Configure only stock, documented ExecuTorch options. CMSIS_NN_LOCAL_PATH is
# the stock Cortex-M provider hook; the local adapter keeps its target creation
# idempotent for either preconfigured NSX provider.
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PORTABLE_OPS "${_nsx_executorch_build_portable_ops}"
    CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_CORTEX_M ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PTHREADPOOL OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_CPUINFO OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_DEVTOOLS "${NSX_EXECUTORCH_ENABLE_PROFILING}"
    CACHE BOOL "" FORCE)
set(EXECUTORCH_ENABLE_LOGGING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_ENABLE_EVENT_TRACER "${NSX_EXECUTORCH_ENABLE_PROFILING}"
    CACHE BOOL "" FORCE)
# The verifier is ~20-45 KB of flash that only executes when Program::load is
# called with Verification::InternalConsistency; the adapter loads with the
# default Verification::Minimal, and NSX deployments SHA-check the PTE via the
# helia-torch sidecar manifest instead. See docs/build-options.md.
set(EXECUTORCH_ENABLE_PROGRAM_VERIFICATION
    "${NSX_EXECUTORCH_ENABLE_PROGRAM_VERIFICATION}" CACHE BOOL "" FORCE)
# Only meaningful under CMAKE_BUILD_TYPE=Release (it edits ExecuTorch's
# CMAKE_*_FLAGS_RELEASE); the authoritative optimization level is the explicit
# per-target flag applied by _nsx_executorch_apply_module_options below, which
# wins regardless of build type. Keep this consistent with the knob anyway.
if(NSX_EXECUTORCH_OPTIMIZATION STREQUAL "size")
  set(EXECUTORCH_OPTIMIZE_SIZE ON CACHE BOOL "" FORCE)
else()
  set(EXECUTORCH_OPTIMIZE_SIZE OFF CACHE BOOL "" FORCE)
endif()
set(EXECUTORCH_USE_DL OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_PAL_DEFAULT minimal CACHE STRING "" FORCE)
set(EXECUTORCH_SELECT_OPS_LIST
    "${NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST}" CACHE STRING "" FORCE)
set(_nsx_executorch_provider_adapter
    "${CMAKE_CURRENT_BINARY_DIR}/cmsis-nn-provider")
file(MAKE_DIRECTORY "${_nsx_executorch_provider_adapter}")
configure_file(
  "${NSX_EXECUTORCH_ROOT}/cmake/cmsis-nn-provider/CMakeLists.txt"
  "${_nsx_executorch_provider_adapter}/CMakeLists.txt"
  COPYONLY)
file(COPY "${_nsx_executorch_cmsis_nn_root}/Include"
     DESTINATION "${_nsx_executorch_provider_adapter}")
set(CMSIS_NN_LOCAL_PATH "${_nsx_executorch_provider_adapter}"
    CACHE PATH "" FORCE)
if(_nsx_executorch_cmsis_root)
  set(CMSIS_PATH "${_nsx_executorch_cmsis_root}" CACHE PATH "" FORCE)
endif()
# Static kernel-registry capacity (12 bytes of .bss per slot on 32-bit).
# Registrations at static-init time: 25 prim ops + 16 cortex_m ops (+5
# cortex_m_ns ops), plus one per entry in
# NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST. Overflow aborts before main() —
# silently, since logging is off and the minimal PAL traps — so keep visible
# headroom: 64 leaves 23 portable slots, 96 leaves 50.
if(NSX_EXECUTORCH_ENABLE_NS_OPS)
  set(MAX_KERNEL_NUM 96 CACHE STRING "" FORCE)
else()
  set(MAX_KERNEL_NUM 64 CACHE STRING "" FORCE)
endif()
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL
    "Never fetch dependencies while configuring nsx-executorch" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

# ExecuTorch v1.3.0's schema rule assumes this output directory already exists.
# Creating a build-tree directory here avoids carrying a source patch.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/schema/include/executorch/schema")

# This repository is a build-tree NSX module, not an installed ExecuTorch SDK.
# Suppress the stock subproject's package/export rules: its cmsis-nn export
# cannot represent an NSX-selected provider and is not consumed by NSX apps.
set(_nsx_executorch_skip_install_rules "${CMAKE_SKIP_INSTALL_RULES}")
set(CMAKE_SKIP_INSTALL_RULES ON)
add_subdirectory(
  "${NSX_EXECUTORCH_SOURCE_ROOT}"
  "${CMAKE_CURRENT_BINARY_DIR}/executorch"
  EXCLUDE_FROM_ALL)
set(CMAKE_SKIP_INSTALL_RULES "${_nsx_executorch_skip_install_rules}")

# ExecuTorch points cmsis-nn's public include directory at its local provider
# hook. Restore the selected provider's actual headers after the adapter has
# made target creation idempotent.
get_target_property(_nsx_executorch_cmsis_nn_type cmsis-nn TYPE)
if(_nsx_executorch_cmsis_nn_type STREQUAL "INTERFACE_LIBRARY")
  target_include_directories(cmsis-nn INTERFACE
    "$<BUILD_INTERFACE:${_nsx_executorch_cmsis_nn_root}/Include>")
else()
  target_include_directories(cmsis-nn PUBLIC
    "$<BUILD_INTERFACE:${_nsx_executorch_cmsis_nn_root}/Include>")
endif()

# ExecuTorch creates concrete libraries that do not inherit the NSX board ABI
# flags. Apply those flags privately across its subproject after target
# creation; the stock source tree remains untouched.
#
# The board interface carries the app's optimization policy (-g -O3
# -ffast-math for GCC/ATfE, -Ofast for armclang) in addition to the ABI
# flags. Compile options land after CMAKE_<LANG>_FLAGS_<CONFIG> on the
# command line and the last -O wins, so applying the raw list would silently
# override every optimization choice made for the ExecuTorch subtree — that
# is exactly what shipped before this audit: the whole runtime built at -O3
# -ffast-math while EXECUTORCH_OPTIMIZE_SIZE=ON suggested -Os. Keep only the
# ABI/codegen-relevant flags and own the optimization level explicitly via
# NSX_EXECUTORCH_OPTIMIZATION. -ffast-math is dropped outright: it changes
# float semantics inside kernels and the runtime, and belongs to app code
# only.
get_target_property(_nsx_executorch_board_options
                    nsx::board_flags INTERFACE_COMPILE_OPTIONS)
if(_nsx_executorch_board_options MATCHES "-NOTFOUND$")
  set(_nsx_executorch_board_options "")
endif()
set(_nsx_executorch_abi_options "")
foreach(_option IN LISTS _nsx_executorch_board_options)
  if(NOT _option MATCHES "^-O" AND NOT _option STREQUAL "-ffast-math")
    list(APPEND _nsx_executorch_abi_options "${_option}")
  endif()
endforeach()

# Both flag sets are carried by INTERFACE targets and applied by *linking*
# them, not by target_compile_options: the CMSIS-NN provider links the board
# interface PUBLIC, so every dependent (cortex_m_kernels, cortex_m_ns_kernels)
# inherits the raw board options through its LINK_LIBRARIES — and usage
# requirements are collected in link order, after a target's own
# COMPILE_OPTIONS. Linking these targets appends our flags after that
# inherited tail, which is the only position where the -O choice and
# -fno-fast-math actually win.
#
# Hygiene flags stock ExecuTorch only applies under CMAKE_BUILD_TYPE=Release
# (it edits CMAKE_*_FLAGS_RELEASE); apply them unconditionally so a consumer
# with no build type set still gets a linkable, section-GC-able runtime.
add_library(nsx_executorch_abi_flags INTERFACE)
target_compile_options(nsx_executorch_abi_flags INTERFACE
    ${_nsx_executorch_abi_options})
add_library(nsx_executorch_module_flags INTERFACE)
# -fno-fast-math cancels the board policy's -ffast-math bundle; the two flags
# after it re-grant only its mild guarantees (no errno, no NaN/Inf operands
# — these kernels run between dequantize/quantize steps, so inputs are scaled
# integer ranges). They do not permit reassociation or other value-changing
# rewrites, but they are what lets GCC vectorize float compare-selects
# (portable leaky_relu was 4x slower without them). Order matters:
# -fno-fast-math resets both, so they must follow it.
target_compile_options(nsx_executorch_module_flags INTERFACE
    ${_nsx_executorch_abi_options}
    ${_nsx_executorch_opt_flag}
    -fno-fast-math
    -fno-math-errno
    -ffinite-math-only
    -ffunction-sections
    -fdata-sections
    -fno-exceptions
    -fno-asynchronous-unwind-tables
    -fno-unwind-tables
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)

# The CMSIS-NN provider pins its own optimization level (-Ofast via
# NSX_CMSIS_NN_OPTIMIZATION / CMSIS_OPTIMIZATION_LEVEL); targets that already
# chose a -O keep it — and their fast-math — and receive only the ABI flags.
function(_nsx_executorch_apply_module_options directory)
  get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_target IN LISTS _targets)
    get_target_property(_target_type "${_target}" TYPE)
    if(NOT _target_type STREQUAL "INTERFACE_LIBRARY"
       AND NOT _target_type STREQUAL "UTILITY")
      get_target_property(_target_options "${_target}" COMPILE_OPTIONS)
      if(_target_options MATCHES "(^|;)-O")
        set(_flags_target nsx_executorch_abi_flags)
      else()
        set(_flags_target nsx_executorch_module_flags)
      endif()
      # Property-level append rather than target_link_libraries: several
      # stock targets use the plain signature, which cannot be mixed with
      # keyword calls from here.
      set_property(TARGET "${_target}" APPEND PROPERTY
                   LINK_LIBRARIES "${_flags_target}")
    endif()
  endforeach()
  get_property(_subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(_subdirectory IN LISTS _subdirectories)
    _nsx_executorch_apply_module_options("${_subdirectory}")
  endforeach()
endfunction()
_nsx_executorch_apply_module_options("${NSX_EXECUTORCH_SOURCE_ROOT}")

if(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns")
  # NS-CMSIS-NN >= 7.2 adds a weight-sum context to three stock wrapper
  # signatures. Force-include a private adapter only for ExecuTorch's C++
  # Cortex-M operators; the provider sources keep their native API.
  target_compile_options(cortex_m_kernels PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${NSX_EXECUTORCH_ROOT}/includes-private/nsx_cmsis_nn_compat.h>")
endif()

if(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "arm"
   AND NOT TARGET nsx::arm_cmsis_nn)
  # Stock ExecuTorch just created the single `cmsis-nn` target above, from
  # arm-cmsis-nn's own vendored source. Now add arm-cmsis-nn's own
  # CMakeLists.txt so it can publish its nsx::arm_cmsis_nn contract; it is
  # written to detect the existing `cmsis-nn` target and skip re-creating it
  # (see `if(NOT TARGET cmsis-nn)` in arm-cmsis-nn/CMakeLists.txt), so this
  # never produces a duplicate target.
  if(NOT DEFINED NSX_BOARD_FLAGS_TARGET)
    set(NSX_BOARD_FLAGS_TARGET nsx::board_flags)
  endif()
  add_subdirectory(
    "${_nsx_executorch_arm_cmsis_nn_root}"
    "${CMAKE_CURRENT_BINARY_DIR}/arm-cmsis-nn")
  if(NOT TARGET nsx::arm_cmsis_nn)
    message(FATAL_ERROR
      "nsx-executorch: arm-cmsis-nn did not export nsx::arm_cmsis_nn.")
  endif()
endif()

if(NSX_EXECUTORCH_ENABLE_NS_OPS)
  # Ambiq ns-cmsis-nn additional kernels: build the out-of-tree cortex_m_ns::
  # operator library using stock ExecuTorch codegen (public extension point;
  # the pinned submodule stays untouched). PTEs containing cortex_m_ns:: ops
  # fail fast at Method load when this block is disabled.
  #
  # Codegen additionally needs PyYAML on the host interpreter (the vendored
  # tools/python tree only covers torchgen + typing_extensions). Fail at
  # configure time with clear guidance instead of at build time.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import yaml"
    RESULT_VARIABLE _nsx_executorch_yaml_result
    OUTPUT_QUIET ERROR_QUIET)
  if(NOT _nsx_executorch_yaml_result EQUAL 0)
    message(FATAL_ERROR
      "nsx-executorch: NSX_EXECUTORCH_ENABLE_NS_OPS needs PyYAML on the host "
      "Python (${NSX_EXECUTORCH_HOST_PYTHON}) for ExecuTorch codegen. "
      "Install it (e.g. 'pip install pyyaml') or select a different "
      "interpreter with -DPython3_EXECUTABLE=...")
  endif()
  set(EXECUTORCH_ROOT "${NSX_EXECUTORCH_SOURCE_ROOT}")
  if(NOT COMMAND gen_selected_ops)
    include("${NSX_EXECUTORCH_SOURCE_ROOT}/tools/cmake/Codegen.cmake")
  endif()

  set(_nsx_executorch_ns_ops_yaml
      "${NSX_EXECUTORCH_ROOT}/ops-ns/operators_ns.yaml")
  gen_selected_ops(
    LIB_NAME "cortex_m_ns_ops_lib"
    OPS_SCHEMA_YAML "${_nsx_executorch_ns_ops_yaml}")
  generate_bindings_for_kernels(
    LIB_NAME "cortex_m_ns_ops_lib"
    CUSTOM_OPS_YAML "${_nsx_executorch_ns_ops_yaml}")

  add_library(cortex_m_ns_kernels STATIC
    "${NSX_EXECUTORCH_ROOT}/ops-ns/op_quantized_sub.cpp"
    "${NSX_EXECUTORCH_ROOT}/ops-ns/op_quantized_hardswish.cpp"
    "${NSX_EXECUTORCH_ROOT}/ops-ns/op_quantized_mean.cpp"
    "${NSX_EXECUTORCH_ROOT}/ops-ns/op_quantized_relu.cpp"
    "${NSX_EXECUTORCH_ROOT}/ops-ns/op_quantized_leaky_relu.cpp")
  target_link_libraries(cortex_m_ns_kernels
    PRIVATE cmsis-nn
    PRIVATE executorch
    PRIVATE kernels_util_all_deps)
  target_include_directories(cortex_m_ns_kernels PRIVATE
    "${NSX_EXECUTORCH_SOURCE_ROOT}/.."
    "${NSX_EXECUTORCH_SOURCE_ROOT}/runtime/core/portable_type/c10")
  target_link_libraries(cortex_m_ns_kernels PRIVATE
                        nsx_executorch_module_flags)

  gen_operators_lib(
    LIB_NAME "cortex_m_ns_ops_lib"
    KERNEL_LIBS cortex_m_ns_kernels
    DEPS executorch)
  target_link_libraries(cortex_m_ns_ops_lib PRIVATE
                        nsx_executorch_module_flags)

  # Mirror the directory-scoped add_definitions() every in-tree ExecuTorch
  # target gets; these two libraries live in consumer scope and would
  # otherwise compile ExecuTorch headers with a different configuration than
  # executorch_core. ET_EVENT_TRACER_ENABLED in particular gates inline
  # tracing code in runtime headers and must agree across every TU.
  foreach(_ns_target cortex_m_ns_kernels cortex_m_ns_ops_lib)
    target_compile_definitions(${_ns_target} PRIVATE
      ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0
      ET_MIN_LOG_LEVEL=Info)
    if(NSX_EXECUTORCH_ENABLE_PROFILING)
      target_compile_definitions(${_ns_target} PRIVATE ET_EVENT_TRACER_ENABLED)
    endif()
  endforeach()
endif()

add_library(nsx_executorch STATIC src/nsx_executorch.cpp)
add_library(nsx::executorch ALIAS nsx_executorch)
set_target_properties(nsx_executorch PROPERTIES EXPORT_NAME executorch)
target_compile_features(nsx_executorch PUBLIC cxx_std_17)
if(NSX_EXECUTORCH_ENABLE_PROFILING)
  # ExecuTorch applies this definition inside its own source directory. The
  # adapter lives one directory above it and must opt into the same ABI path.
  target_compile_definitions(nsx_executorch PRIVATE ET_EVENT_TRACER_ENABLED)
endif()
target_include_directories(nsx_executorch PUBLIC
  $<BUILD_INTERFACE:${NSX_EXECUTORCH_ROOT}/includes-api>)
target_link_libraries(nsx_executorch PUBLIC
  nsx::core
  nsx::cmsis_core
  nsx::soc_hal)
target_link_libraries(nsx_executorch PRIVATE
  executorch
  extension_data_loader
  cortex_m_ops_lib
  cortex_m_kernels)
if(NSX_EXECUTORCH_ENABLE_NS_OPS)
  target_link_libraries(nsx_executorch PRIVATE
    cortex_m_ns_ops_lib
    cortex_m_ns_kernels)
endif()
if(_nsx_executorch_build_portable_ops)
  target_link_libraries(nsx_executorch PRIVATE executorch_kernels)
endif()

# The adapter is created in consumer scope, outside the subtree walk above,
# and none of its linked nsx:: contracts are required to carry the board ABI
# flags. In a bootstrapped NSX app the app-level flags cover it; standalone
# consumers (like tests/real-provider) would otherwise compile this one TU
# for the host default architecture. Linked last so these options land after
# any board interface inherited through the nsx:: contracts.
target_link_libraries(nsx_executorch PRIVATE nsx_executorch_module_flags)

# --- Toolchain-specific accommodations (issue #6) ---
# All per-toolchain branching lives here, in one place. ATfE (upstream LLVM
# clang, CMAKE_CXX_COMPILER_ID "Clang") is the only non-GCC toolchain
# currently supported; armclang identifies as "ARMClang" and is out of scope.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # ATfE's bare-metal libc++ is built without std::random_device (no entropy
  # source on the target); GCC's libstdc++ provides one unconditionally,
  # which is why the arm-none-eabi-gcc build never noticed. portable_kernels
  # compiles every portable source regardless of EXECUTORCH_SELECT_OPS_LIST
  # (selection only controls registration; --gc-sections strips the rest), so
  # the three kernels that construct a std::random_device break the ATfE
  # build even when their ops are never registered. Drop those sources, and
  # fail loudly if a select list actually asks for them.
  if(_nsx_executorch_build_portable_ops)
    if(NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST MATCHES
       "aten::(rand|randn|native_dropout)\\.")
      message(FATAL_ERROR
        "nsx-executorch: aten::rand/randn/native_dropout portable kernels "
        "require std::random_device, which this toolchain's libc++ does not "
        "provide on bare metal. Remove them from "
        "NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST or build with "
        "arm-none-eabi-gcc.")
    endif()
    get_target_property(_nsx_executorch_portable_sources
                        portable_kernels SOURCES)
    list(FILTER _nsx_executorch_portable_sources
         EXCLUDE REGEX "op_(rand|randn|native_dropout)\\.cpp$")
    set_target_properties(portable_kernels PROPERTIES
      SOURCES "${_nsx_executorch_portable_sources}")
  endif()

  # The adapter includes am_mcu_apollo.h as C++. AmbiqSuite's HAL detects the
  # GCC toolchain via __GNUC_STDC_INLINE__, which clang defines in C mode but
  # not in C++ (armclang is matched via __ARMCC_VERSION instead), so C++
  # inclusion hits "#error Compiler is unknown". Define it for this target
  # only; the HAL's guarded declarations are attribute-compatible with clang,
  # and the corresponding definitions live in HAL C sources where clang
  # defines the macro natively. The reserved-literal suppression mirrors what
  # nsx_toolchain_flags.cmake already applies for armclang against the same
  # headers (C-style literal concatenation without a space).
  target_compile_definitions(nsx_executorch PRIVATE __GNUC_STDC_INLINE__=1)
  target_compile_options(nsx_executorch PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-reserved-user-defined-literal>)
endif()
