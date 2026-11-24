# Count the kernels nsx_executorch registers at static-init time, from the
# sources that register them, and fail the build if MAX_KERNEL_NUM is smaller.
#
# ExecuTorch's kernel registry is a fixed table of MAX_KERNEL_NUM slots. A
# registration that does not fit aborts before main(), with no output (logging
# is off and the minimal PAL traps), so the count must be right before the
# firmware is linked. CMakeLists.txt checks an estimate at configure time; this
# script checks the exact figure once codegen has written the registration
# sources: every `Kernel(` entry of each generated
# RegisterCodegenUnboxedKernelsEverything.cpp, plus the prim ops of
# kernels/prim_ops/register_prim_ops.cpp.
#
# Script mode:
#   cmake -DREGISTRATION_SOURCES=<a.cpp;b.cpp> -DPRIM_OPS_SOURCE=<cpp>
#         -DMAX_KERNEL_NUM=<n> [-DSTAMP=<file>] -P nsx_executorch_kernel_count.cmake
#
# Also included by CMakeLists.txt for nsx_executorch_count_kernel_entries().

# The number of `Kernel(` entries in a registration source: one per line, as
# ExecuTorch's codegen and register_prim_ops.cpp write them. A file with none
# is an error, since a source is only counted when it registers something.
function(nsx_executorch_count_kernel_entries out_var source)
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "nsx-executorch: kernel registration source ${source} does not exist.")
  endif()
  file(STRINGS "${source}" _entries REGEX "^[ \t]*Kernel\\(")
  list(LENGTH _entries _count)
  if(_count EQUAL 0)
    message(FATAL_ERROR
      "nsx-executorch: no `Kernel(` entries in ${source}; its layout changed, "
      "so the kernel count cannot be checked. Update "
      "cmake/nsx_executorch_kernel_count.cmake.")
  endif()
  set(${out_var} ${_count} PARENT_SCOPE)
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  foreach(_required PRIM_OPS_SOURCE MAX_KERNEL_NUM)
    if(NOT DEFINED ${_required})
      message(FATAL_ERROR "nsx_executorch_kernel_count.cmake: ${_required} is required")
    endif()
  endforeach()
  nsx_executorch_count_kernel_entries(_total "${PRIM_OPS_SOURCE}")
  set(_detail "${_total} prim ops")
  foreach(_source IN LISTS REGISTRATION_SOURCES)
    nsx_executorch_count_kernel_entries(_count "${_source}")
    math(EXPR _total "${_total} + ${_count}")
    get_filename_component(_lib "${_source}" DIRECTORY)
    get_filename_component(_lib "${_lib}" NAME)
    string(APPEND _detail ", ${_count} ${_lib}")
  endforeach()
  if(_total GREATER MAX_KERNEL_NUM)
    message(FATAL_ERROR
      "nsx-executorch: ${_total} kernels are registered (${_detail}) but "
      "MAX_KERNEL_NUM is ${MAX_KERNEL_NUM}; the overflow would abort silently "
      "before main(). Shorten NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST or select "
      "operators with NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST / "
      "NSX_EXECUTORCH_NS_SELECT_OPS_LIST.")
  endif()
  message(STATUS "nsx-executorch: ${_total} of ${MAX_KERNEL_NUM} kernel registry slots used (${_detail})")
  if(DEFINED STAMP)
    file(WRITE "${STAMP}" "${_total}\n")
  endif()
endif()
