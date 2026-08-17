cmake_minimum_required(VERSION 3.19)

if(TARGET cmsis-nn)
  return()
endif()

if(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "arm")
  add_subdirectory(
    "${_nsx_executorch_cmsis_nn_root}"
    "${CMAKE_CURRENT_BINARY_DIR}/arm-source"
    EXCLUDE_FROM_ALL)
elseif(NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns")
  if(TARGET nsx::cmsis_nn)
    add_library(cmsis-nn INTERFACE)
    target_link_libraries(cmsis-nn INTERFACE
      "$<BUILD_INTERFACE:nsx::cmsis_nn>")
  else()
    add_subdirectory(
      "${_nsx_executorch_cmsis_nn_root}"
      "${CMAKE_CURRENT_BINARY_DIR}/ns-source"
      EXCLUDE_FROM_ALL)
  endif()
endif()

if(NOT TARGET cmsis-nn)
  message(FATAL_ERROR
    "nsx-executorch: the selected provider did not create cmsis-nn.")
endif()
