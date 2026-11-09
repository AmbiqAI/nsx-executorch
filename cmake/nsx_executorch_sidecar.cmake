# Configure nsx-executorch from the sidecars helia-torch writes next to each
# PTE (<model>.pte.json), so the firmware builds what the exported models need.
#
# The export already knows which operators stay in ExecuTorch (after any
# heliaAOT delegation) and which heliaAOT modules the program calls; the
# sidecar records both. These functions turn that into the build, so nobody
# copies operator lists, options or module names into CMake by hand.
#
#   include(<nsx-executorch>/cmake/nsx_executorch_sidecar.cmake)
#
#   # Before nsx-executorch is configured (before nsx_bootstrap_app(), or
#   # before add_subdirectory(<nsx-executorch>)):
#   nsx_executorch_configure_from_sidecars(build/kws.pte.json build/vww.pte.json)
#
#   # After it, once the app target exists:
#   nsx_executorch_link_helia_aot_modules(app
#       MODULES_DIR build/helia_aot)    # omit when nsx.yml lists the modules
#
# Several sidecars configure the union of what they need, so firmware can swap
# between a fixed set of models.
#
# Copyright 2026 Ambiq. All Rights Reserved.

include_guard(GLOBAL)

if(CMAKE_VERSION VERSION_LESS 3.19)
  message(FATAL_ERROR
    "nsx-executorch: reading PTE sidecars needs CMake 3.19 or later (string(JSON)).")
endif()

# Append the JSON string array at <json> <key...> to the list <out_var>.
function(_nsx_executorch_sidecar_strings out_var json)
  set(_values "${${out_var}}")
  string(JSON _count ERROR_VARIABLE _err LENGTH "${json}" ${ARGN})
  if(_err)
    string(REPLACE ";" "." _key "${ARGN}")
    message(FATAL_ERROR "nsx-executorch: sidecar has no ${_key} list (${_err}).")
  endif()
  if(_count GREATER 0)
    math(EXPR _last "${_count} - 1")
    foreach(_i RANGE ${_last})
      string(JSON _value GET "${json}" ${ARGN} ${_i})
      list(APPEND _values "${_value}")
    endforeach()
  endif()
  set(${out_var} "${_values}" PARENT_SCOPE)
endfunction()

# Split a comma-separated select list (as passed with -D) into <out_var>,
# dropping blanks.
function(_nsx_executorch_sidecar_split out_var value)
  set(_items "")
  string(REPLACE "," ";" _parts "${value}")
  foreach(_part IN LISTS _parts)
    string(STRIP "${_part}" _part)
    if(NOT _part STREQUAL "")
      list(APPEND _items "${_part}")
    endif()
  endforeach()
  set(${out_var} "${_items}" PARENT_SCOPE)
endfunction()

# nsx_executorch_configure_from_sidecars(<sidecar.pte.json>...)
#
# Sets, in the caller's scope, the nsx-executorch options the sidecars call for.
# They are normal variables, so they take precedence over the cache entries of
# the same name and follow the sidecars on every configure:
#
#   NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST  union of operators.portable
#   NSX_EXECUTORCH_ENABLE_NS_OPS             ON when a model needs cortex_m_ns:: operators
#   NSX_EXECUTORCH_ENABLE_HELIA_AOT          ON when a model calls heliaAOT modules
#   NSX_EXECUTORCH_HELIA_AOT_MODULES         the heliaAOT module names, for
#                                            nsx_executorch_link_helia_aot_modules()
#   NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS      raised to the number of modules when
#                                            there are more than it allows
#   NSX_EXECUTORCH_CMSIS_NN_PROVIDER         ns, when unset and a model needs
#                                            cortex_m_ns:: operators or heliaAOT
#
# A portable list given in the cache (-D...=a,b) is added to the sidecars'
# union, so a build can register extra fallbacks. Options are only ever
# switched on and lists only grow: an option already ON stays ON, and a second
# call (one per model, say) adds to what the first set.
#
# A model that needs cortex_m_ns:: operators or heliaAOT needs the ns provider:
# a provider already set to something else fails here, naming the sidecar.
#
# When the PTE ("<model>.pte" for "<model>.pte.json") is next to the sidecar,
# its SHA-256 must match the one the sidecar records, so a stale sidecar fails
# here instead of at Method::load on the target. Each sidecar, and its PTE
# when it is there, is a configure dependency: re-exporting a model
# reconfigures the build.
function(nsx_executorch_configure_from_sidecars)
  # A sidecar list that expands to nothing (an unset variable, even quoted)
  # must not configure a firmware without the models' operators.
  set(_sidecars "")
  foreach(_sidecar IN LISTS ARGN)
    if(NOT _sidecar STREQUAL "")
      list(APPEND _sidecars "${_sidecar}")
    endif()
  endforeach()
  if(NOT _sidecars)
    message(FATAL_ERROR "nsx_executorch_configure_from_sidecars: no sidecars given.")
  endif()

  # Set here, not at include: include_guard(GLOBAL) runs the file once, and a
  # variable it set would be missing in every other scope.
  set(_schema_read "nsx-executorch.pte-manifest/1")
  set(_portable "")
  set(_modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}")
  set(_needs_ns_ops OFF)
  set(_needs_helia_aot OFF)
  set(_needs_ns_provider "")

  foreach(_sidecar IN LISTS _sidecars)
    get_filename_component(_sidecar "${_sidecar}" ABSOLUTE)
    if(NOT EXISTS "${_sidecar}")
      message(FATAL_ERROR "nsx-executorch: sidecar ${_sidecar} does not exist.")
    endif()
    if(NOT CMAKE_SCRIPT_MODE_FILE)
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_sidecar}")
    endif()
    file(READ "${_sidecar}" _json)

    string(JSON _schema ERROR_VARIABLE _err GET "${_json}" schema)
    if(_err OR NOT _schema STREQUAL _schema_read)
      message(FATAL_ERROR
        "nsx-executorch: ${_sidecar} is not a PTE sidecar this version reads "
        "(schema '${_schema}', expected '${_schema_read}').")
    endif()

    # Only "<model>.pte.json" names its PTE; a sidecar under another name is
    # read without the check.
    string(REGEX REPLACE "\\.pte\\.json$" ".pte" _pte "${_sidecar}")
    if(NOT _pte STREQUAL _sidecar AND EXISTS "${_pte}" AND NOT IS_DIRECTORY "${_pte}")
      if(NOT CMAKE_SCRIPT_MODE_FILE)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_pte}")
      endif()
      file(SHA256 "${_pte}" _actual)
      string(JSON _recorded GET "${_json}" pte_sha256)
      if(NOT _actual STREQUAL _recorded)
        message(FATAL_ERROR
          "nsx-executorch: ${_sidecar} describes a PTE with SHA-256 ${_recorded}, "
          "but ${_pte} is ${_actual}. Re-export the model.")
      endif()
    endif()

    _nsx_executorch_sidecar_strings(_portable "${_json}" operators portable)

    string(JSON _requires_ns GET "${_json}" requires_ns_ops)
    if(_requires_ns)
      set(_needs_ns_ops ON)
      set(_needs_ns_provider "${_sidecar}")
    endif()

    # Sidecars written before the heliaAOT delegate have neither key. Any
    # other sidecar that does not list its modules is malformed: reading it as
    # "no modules" would build a firmware the PTE cannot load on.
    string(JSON _requires_helia_aot ERROR_VARIABLE _no_flag GET "${_json}" requires_helia_aot)
    string(JSON _modules_type ERROR_VARIABLE _no_modules TYPE "${_json}" helia_aot_modules)
    if(_no_modules)
      set(_module_count 0)
      if(NOT _no_flag AND _requires_helia_aot)
        message(FATAL_ERROR
          "nsx-executorch: ${_sidecar} says the program needs heliaAOT but lists no "
          "helia_aot_modules. Re-export the model.")
      endif()
    elseif(NOT _modules_type STREQUAL "ARRAY")
      message(FATAL_ERROR
        "nsx-executorch: ${_sidecar} has a helia_aot_modules that is not a list "
        "(${_modules_type}). Re-export the model.")
    else()
      string(JSON _module_count LENGTH "${_json}" helia_aot_modules)
      if(_module_count EQUAL 0 AND NOT _no_flag AND _requires_helia_aot)
        message(FATAL_ERROR
          "nsx-executorch: ${_sidecar} says the program needs heliaAOT but lists no "
          "helia_aot_modules. Re-export the model.")
      endif()
    endif()
    if(_module_count GREATER 0)
      set(_needs_helia_aot ON)
      set(_needs_ns_provider "${_sidecar}")
      math(EXPR _last "${_module_count} - 1")
      foreach(_i RANGE ${_last})
        string(JSON _module ERROR_VARIABLE _err GET "${_json}" helia_aot_modules ${_i} module)
        if(_err OR _module STREQUAL "")
          message(FATAL_ERROR
            "nsx-executorch: ${_sidecar} has a helia_aot_modules entry without a module "
            "name. Re-export the model.")
        endif()
        list(APPEND _modules "${_module}")
      endforeach()
    endif()
  endforeach()

  if(_needs_ns_provider)
    if(NOT DEFINED NSX_EXECUTORCH_CMSIS_NN_PROVIDER)
      set(NSX_EXECUTORCH_CMSIS_NN_PROVIDER ns PARENT_SCOPE)
    elseif(NOT NSX_EXECUTORCH_CMSIS_NN_PROVIDER STREQUAL "ns")
      message(FATAL_ERROR
        "nsx-executorch: ${_needs_ns_provider} needs cortex_m_ns:: operators or heliaAOT, "
        "which need NSX_EXECUTORCH_CMSIS_NN_PROVIDER=ns, but the provider is "
        "'${NSX_EXECUTORCH_CMSIS_NN_PROVIDER}'.")
    endif()
  endif()

  # Extra portable operators given with -D are registered too.
  _nsx_executorch_sidecar_split(_extra "${NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST}")
  list(APPEND _portable ${_extra})
  list(REMOVE_DUPLICATES _portable)
  list(SORT _portable)
  list(REMOVE_DUPLICATES _modules)

  string(REPLACE ";" "," _portable_list "${_portable}")
  set(NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST "${_portable_list}" PARENT_SCOPE)
  if(_portable_list)
    message(STATUS "nsx-executorch: portable operators from sidecars: ${_portable_list}")
  endif()

  if(_needs_ns_ops)
    set(NSX_EXECUTORCH_ENABLE_NS_OPS ON PARENT_SCOPE)
  endif()
  if(_needs_helia_aot)
    set(NSX_EXECUTORCH_ENABLE_HELIA_AOT ON PARENT_SCOPE)
    string(REPLACE ";" ", " _module_text "${_modules}")
    message(STATUS "nsx-executorch: heliaAOT modules from sidecars: ${_module_text}")
  endif()
  set(NSX_EXECUTORCH_HELIA_AOT_MODULES "${_modules}" PARENT_SCOPE)

  # The registry holds 8 modules unless told otherwise (nsx_helia_aot.h).
  list(LENGTH _modules _module_total)
  set(_max_models 8)
  if(NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS MATCHES "^[1-9][0-9]*$")
    set(_max_models "${NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS}")
  endif()
  if(_module_total GREATER _max_models)
    set(NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS "${_module_total}" PARENT_SCOPE)
    message(STATUS "nsx-executorch: heliaAOT module registry sized for ${_module_total} modules")
  endif()
endfunction()

# nsx_executorch_link_helia_aot_modules(<target> [MODULES_DIR <dir>] [MODULES <name>...])
#
# Links the heliaAOT modules the sidecars name (NSX_EXECUTORCH_HELIA_AOT_MODULES,
# or MODULES) into <target>. A module that is not in the build yet is added
# from MODULES_DIR/<name> (the export's helia_aot output_dir); in an NSX app
# that lists the modules in nsx.yml they are already there. A module found in
# neither place fails the configure, not Method::load on the target.
#
# Call it after nsx-executorch and the ns-cmsis-nn provider are configured:
# the modules link nsx::cmsis_nn and their registration files include
# nsx-executorch's nsx_helia_aot.h. A module added from MODULES_DIR links the
# board's flags through NSX_BOARD_FLAGS_TARGET, which nsx_bootstrap_app() sets
# only inside its own scope: when it is not defined here, nsx::board_flags is
# used, as nsx-executorch itself does.
#
# MODULES given with no names links nothing (a target that takes no modules);
# without MODULES the sidecars' modules are linked.
function(nsx_executorch_link_helia_aot_modules target)
  cmake_parse_arguments(_arg "" "MODULES_DIR" "MODULES" ${ARGN})
  if(_arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "nsx_executorch_link_helia_aot_modules: unexpected arguments ${_arg_UNPARSED_ARGUMENTS}.")
  endif()
  if(NOT TARGET ${target})
    message(FATAL_ERROR "nsx_executorch_link_helia_aot_modules: no target ${target}.")
  endif()
  # cmake_parse_arguments leaves MODULES undefined when it is given no values.
  if("MODULES" IN_LIST ARGN)
    set(_modules "${_arg_MODULES}")
  else()
    set(_modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}")
  endif()
  if(_modules AND NOT TARGET nsx::executorch)
    message(FATAL_ERROR
      "nsx_executorch_link_helia_aot_modules: configure nsx-executorch first.")
  endif()

  foreach(_module IN LISTS _modules)
    string(REPLACE "-" "_" _target "${_module}")
    if(NOT TARGET ${_target} AND _arg_MODULES_DIR)
      get_filename_component(_dir "${_arg_MODULES_DIR}/${_module}" ABSOLUTE)
      if(EXISTS "${_dir}/CMakeLists.txt")
        if(NOT DEFINED NSX_BOARD_FLAGS_TARGET)
          if(NOT TARGET nsx::board_flags)
            message(FATAL_ERROR
              "nsx_executorch_link_helia_aot_modules: heliaAOT module ${_module} links the "
              "board's flags, but NSX_BOARD_FLAGS_TARGET is not set and there is no "
              "nsx::board_flags target. Call this after nsx_bootstrap_app().")
          endif()
          set(NSX_BOARD_FLAGS_TARGET nsx::board_flags)
        endif()
        add_subdirectory("${_dir}" "${CMAKE_BINARY_DIR}/_nsx_executorch_helia_aot/${_module}")
      endif()
    endif()
    if(NOT TARGET ${_target})
      if(_arg_MODULES_DIR)
        set(_where "is neither in the build nor in ${_arg_MODULES_DIR}")
      else()
        set(_where "is not in the build, and no MODULES_DIR was given")
      endif()
      message(FATAL_ERROR
        "nsx-executorch: the program calls heliaAOT module ${_module}, which "
        "${_where}. List it in nsx.yml after nsx-cmsis-nn, or pass MODULES_DIR "
        "<the export's helia_aot output_dir>.")
    endif()
    target_link_libraries(${target} PRIVATE ${_target})
  endforeach()
endfunction()
