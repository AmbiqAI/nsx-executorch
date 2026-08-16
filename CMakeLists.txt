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
set(NSX_EXECUTORCH_NS_CMSIS_NN_ROOT "" CACHE PATH
    "Path to a source-compatible ns-cmsis-nn checkout")
set(NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST "" CACHE STRING
    "Comma-separated portable ATen fallback operators; empty disables them")
set(NSX_EXECUTORCH_ENABLE_PROFILING OFF CACHE BOOL
    "Enable EventTracer-backed per-operator callbacks")

if(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "arm")
  set(_nsx_executorch_cmsis_nn_root
      "${NSX_EXECUTORCH_ROOT}/external/CMSIS-NN")
  set(_nsx_executorch_cmsis_root
      "${NSX_EXECUTORCH_ROOT}/external/CMSIS_6")
elseif(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns")
  if(NOT NSX_EXECUTORCH_NS_CMSIS_NN_ROOT)
    set(_nsx_executorch_ns_candidates
        "${NSX_EXECUTORCH_ROOT}/../ns-cmsis-nn"
        "${NSX_EXECUTORCH_ROOT}/external/ns-cmsis-nn")
    foreach(_candidate IN LISTS _nsx_executorch_ns_candidates)
      if(EXISTS "${_candidate}/CMakeLists.txt")
        get_filename_component(NSX_EXECUTORCH_NS_CMSIS_NN_ROOT
            "${_candidate}" ABSOLUTE)
        break()
      endif()
    endforeach()
  endif()
  if(NOT EXISTS "${NSX_EXECUTORCH_NS_CMSIS_NN_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
      "nsx-executorch: provider=ns requires a source-compatible "
      "ns-cmsis-nn checkout. Set NSX_EXECUTORCH_NS_CMSIS_NN_ROOT.")
  endif()
  set(_nsx_executorch_cmsis_nn_root
      "${NSX_EXECUTORCH_NS_CMSIS_NN_ROOT}")
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
# the stock Cortex-M provider hook; both arm and ns providers must satisfy the
# same source and target contract.
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
set(EXECUTORCH_ENABLE_PROGRAM_VERIFICATION ON CACHE BOOL "" FORCE)
set(EXECUTORCH_OPTIMIZE_SIZE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_USE_DL OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_PAL_DEFAULT minimal CACHE STRING "" FORCE)
set(EXECUTORCH_SELECT_OPS_LIST
    "${NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST}" CACHE STRING "" FORCE)
set(CMSIS_NN_LOCAL_PATH "${_nsx_executorch_cmsis_nn_root}"
    CACHE PATH "" FORCE)
if(_nsx_executorch_cmsis_root)
  set(CMSIS_PATH "${_nsx_executorch_cmsis_root}" CACHE PATH "" FORCE)
endif()
set(MAX_KERNEL_NUM 64 CACHE STRING "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

# ExecuTorch v1.3.0's schema rule assumes this output directory already exists.
# Creating a build-tree directory here avoids carrying a source patch.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/schema/include/executorch/schema")

add_subdirectory(
  "${NSX_EXECUTORCH_SOURCE_ROOT}"
  "${CMAKE_CURRENT_BINARY_DIR}/executorch"
  EXCLUDE_FROM_ALL)

# ExecuTorch creates concrete libraries that do not inherit the NSX board ABI
# flags. Apply those flags privately across its subproject after target
# creation; the stock source tree remains untouched.
get_target_property(_nsx_executorch_board_options
                    nsx::board_flags INTERFACE_COMPILE_OPTIONS)
if(_nsx_executorch_board_options MATCHES "-NOTFOUND$")
  set(_nsx_executorch_board_options "")
endif()
function(_nsx_executorch_apply_board_options directory)
  get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_target IN LISTS _targets)
    get_target_property(_target_type "${_target}" TYPE)
    if(NOT _target_type STREQUAL "INTERFACE_LIBRARY"
       AND NOT _target_type STREQUAL "UTILITY")
      target_compile_options("${_target}" PRIVATE
                             ${_nsx_executorch_board_options})
    endif()
  endforeach()
  get_property(_subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(_subdirectory IN LISTS _subdirectories)
    _nsx_executorch_apply_board_options("${_subdirectory}")
  endforeach()
endfunction()
_nsx_executorch_apply_board_options("${NSX_EXECUTORCH_SOURCE_ROOT}")

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
if(_nsx_executorch_build_portable_ops)
  target_link_libraries(nsx_executorch PRIVATE executorch_kernels)
endif()
