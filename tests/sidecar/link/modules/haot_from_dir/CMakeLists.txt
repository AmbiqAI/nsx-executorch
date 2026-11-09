# Stand-in for a heliaAOT module that is not in the build yet, so
# nsx_executorch_link_helia_aot_modules() adds it from MODULES_DIR. It carries
# the guard of a real module (heliaAOT's nsx CMakeLists template).
if(NOT DEFINED NSX_BOARD_FLAGS_TARGET)
  message(FATAL_ERROR "haot_from_dir: NSX_BOARD_FLAGS_TARGET is not defined.")
endif()
add_library(haot_from_dir INTERFACE)
target_link_libraries(haot_from_dir INTERFACE ${NSX_BOARD_FLAGS_TARGET})
