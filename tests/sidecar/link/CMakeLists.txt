# Configure-only test of nsx_executorch_link_helia_aot_modules():
#
#   cmake -S tests/sidecar/link -B build/sidecar-link             # must succeed
#   cmake -S tests/sidecar/link -B build/sidecar-link-missing \
#         -DNSX_EXECUTORCH_TEST_MISSING=ON                        # must fail
#
# nsx-executorch and the heliaAOT modules are stand-ins: what is tested is
# which modules get added and linked, and that a missing one fails configure.
cmake_minimum_required(VERSION 3.24)
project(nsx_executorch_sidecar_link LANGUAGES C)

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/nsx_executorch_sidecar.cmake")

add_library(nsx_executorch INTERFACE)
add_library(nsx::executorch ALIAS nsx_executorch)
add_library(haot_in_build INTERFACE)  # e.g. listed in nsx.yml
# nsx_bootstrap_app() creates the board flags target but sets
# NSX_BOARD_FLAGS_TARGET only inside its own scope, so it is not defined here.
add_library(nsx_board_flags INTERFACE)
add_library(nsx::board_flags ALIAS nsx_board_flags)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/app.c" "int main(void) { return 0; }\n")
add_executable(app "${CMAKE_CURRENT_BINARY_DIR}/app.c")

# As nsx_executorch_configure_from_sidecars() leaves it.
set(NSX_EXECUTORCH_HELIA_AOT_MODULES haot_in_build haot_from_dir)
nsx_executorch_link_helia_aot_modules(app
    MODULES_DIR "${CMAKE_CURRENT_LIST_DIR}/modules")

get_target_property(_linked app LINK_LIBRARIES)
if(NOT "${_linked}" STREQUAL "haot_in_build;haot_from_dir")
  message(FATAL_ERROR "app links '${_linked}', expected haot_in_build;haot_from_dir")
endif()

# MODULES with no names is a target that takes no modules, not "use the sidecars'".
add_executable(tool "${CMAKE_CURRENT_BINARY_DIR}/app.c")
set(_none "")
nsx_executorch_link_helia_aot_modules(tool MODULES ${_none})
get_target_property(_linked tool LINK_LIBRARIES)
if(_linked)
  message(FATAL_ERROR "tool links '${_linked}', expected nothing")
endif()

if(NSX_EXECUTORCH_TEST_MISSING)
  nsx_executorch_link_helia_aot_modules(app MODULES haot_missing
      MODULES_DIR "${CMAKE_CURRENT_LIST_DIR}/modules")
endif()
