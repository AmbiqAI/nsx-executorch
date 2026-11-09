# Print the options nsx_executorch_configure_from_sidecars() sets for the
# sidecars in SIDECARS (a ;-separated list), one NAME=value line each. The
# aot tests run it on sidecars that a real export wrote.
#
#   cmake -DSIDECARS=a.pte.json -P tests/sidecar/print_config.cmake
#
# -DQUOTED=ON passes the list as one quoted argument, as an app that writes
# nsx_executorch_configure_from_sidecars("${MODEL_SIDECARS}") does.

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/nsx_executorch_sidecar.cmake")

if(QUOTED)
  nsx_executorch_configure_from_sidecars("${SIDECARS}")
else()
  nsx_executorch_configure_from_sidecars(${SIDECARS})
endif()
foreach(_var
    NSX_EXECUTORCH_CMSIS_NN_PROVIDER
    NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS
    NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST
    NSX_EXECUTORCH_ENABLE_NS_OPS
    NSX_EXECUTORCH_ENABLE_HELIA_AOT
    NSX_EXECUTORCH_HELIA_AOT_MODULES)
  message(NOTICE "${_var}=${${_var}}")
endforeach()
