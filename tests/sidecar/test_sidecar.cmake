# Tests for cmake/nsx_executorch_sidecar.cmake, run in CMake script mode:
#
#   cmake -P tests/sidecar/test_sidecar.cmake
#
# Each case writes sidecars into a scratch directory, configures from them and
# checks the variables that result. Error cases run in a child cmake process,
# which must fail with the expected message.

set(_helper "${CMAKE_CURRENT_LIST_DIR}/../../cmake/nsx_executorch_sidecar.cmake")
include("${_helper}")

if(NOT WORK)
  set(WORK "${CMAKE_CURRENT_BINARY_DIR}/sidecar-test")
endif()
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

# write_sidecar(<name> [PTE <bytes>] [SHA <sha>] [CORTEX_M ..] [NS ..] [PORTABLE ..] [MODULES ..] [SCHEMA s])
# writes <WORK>/<name>.pte.json, and <WORK>/<name>.pte when PTE is given.
function(write_sidecar name)
  cmake_parse_arguments(_a "NO_MODULES_KEY" "PTE;SHA;SCHEMA" "CORTEX_M;NS;PORTABLE;MODULES" ${ARGN})
  if(NOT DEFINED _a_SCHEMA)
    set(_a_SCHEMA "nsx-executorch.pte-manifest/1")
  endif()
  set(_sha "${_a_SHA}")
  if(DEFINED _a_PTE)
    file(WRITE "${WORK}/${name}.pte" "${_a_PTE}")
    if(NOT _sha)
      file(SHA256 "${WORK}/${name}.pte" _sha)
    endif()
  endif()
  set(_json "{}")
  string(JSON _json SET "${_json}" schema "\"${_a_SCHEMA}\"")
  string(JSON _json SET "${_json}" pte_sha256 "\"${_sha}\"")
  set(_ops "{}")
  foreach(_kind CORTEX_M NS PORTABLE)
    set(_array "[]")
    set(_i 0)
    foreach(_op IN LISTS _a_${_kind})
      string(JSON _array SET "${_array}" ${_i} "\"${_op}\"")
      math(EXPR _i "${_i} + 1")
    endforeach()
    if(_kind STREQUAL "CORTEX_M")
      set(_key cortex_m)
    elseif(_kind STREQUAL "NS")
      set(_key cortex_m_ns)
    else()
      set(_key portable)
    endif()
    string(JSON _ops SET "${_ops}" ${_key} "${_array}")
  endforeach()
  string(JSON _json SET "${_json}" operators "${_ops}")
  if(_a_NS)
    string(JSON _json SET "${_json}" requires_ns_ops true)
  else()
    string(JSON _json SET "${_json}" requires_ns_ops false)
  endif()
  if(NOT _a_NO_MODULES_KEY)
    set(_modules "[]")
    set(_i 0)
    foreach(_module IN LISTS _a_MODULES)
      string(JSON _modules SET "${_modules}" ${_i} "{\"module\": \"${_module}\"}")
      math(EXPR _i "${_i} + 1")
    endforeach()
    string(JSON _json SET "${_json}" helia_aot_modules "${_modules}")
    if(_a_MODULES)
      string(JSON _json SET "${_json}" requires_helia_aot true)
    else()
      string(JSON _json SET "${_json}" requires_helia_aot false)
    endif()
  endif()
  file(WRITE "${WORK}/${name}.pte.json" "${_json}")
endfunction()

function(_fail text)
  message(NOTICE "FAIL ${text}")
  set_property(GLOBAL APPEND PROPERTY sidecar_test_failures "${text}")
endfunction()

function(expect name actual expected)
  if(NOT "${actual}" STREQUAL "${expected}")
    _fail("${name}: expected '${expected}', got '${actual}'")
  endif()
endfunction()

# expect_failure(<case> <message regex> <sidecar>... [-D<var>=<value>...])
function(expect_failure case regex)
  set(_sidecars "")
  set(_defines "")
  foreach(_arg IN LISTS ARGN)
    if(_arg MATCHES "^-D")
      list(APPEND _defines "${_arg}")
    else()
      list(APPEND _sidecars "${_arg}")
    endif()
  endforeach()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" "-DSIDECARS=${_sidecars}" ${_defines}
            -P "${CMAKE_CURRENT_LIST_DIR}/print_config.cmake"
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(_rc EQUAL 0)
    _fail("${case}: configure succeeded, expected an error matching '${regex}'")
  else()
    # CMake wraps long messages; match against the text on one line.
    string(REGEX REPLACE "[ \t\r\n]+" " " _text "${_out}${_err}")
  endif()
  if(NOT _rc EQUAL 0 AND NOT _text MATCHES "${regex}")
    _fail("${case}: error did not match '${regex}':\n${_err}")
  endif()
endfunction()

# -- a fully delegated model: only q/dq stay in ExecuTorch ---------------------
function(case_delegated)
  write_sidecar(kws PTE "kws program"
    CORTEX_M "cortex_m::quantize_per_tensor.out" "cortex_m::dequantize_per_tensor.out"
    MODULES haot_0123456789)
  nsx_executorch_configure_from_sidecars("${WORK}/kws.pte.json")
  expect(delegated.cortex_m "${NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST}"
         "cortex_m::dequantize_per_tensor.out,cortex_m::quantize_per_tensor.out")
  expect(delegated.ns "${NSX_EXECUTORCH_NS_SELECT_OPS_LIST}" NONE)
  expect(delegated.helia_aot "${NSX_EXECUTORCH_ENABLE_HELIA_AOT}" ON)
  expect(delegated.ns_ops "${NSX_EXECUTORCH_ENABLE_NS_OPS}" "")
  expect(delegated.portable "${NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST}" "")
  expect(delegated.modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}" haot_0123456789)
endfunction()
case_delegated()

# -- two models: union of operators and modules, NS ops when one needs them ----
function(case_union)
  write_sidecar(a PTE "a"
    CORTEX_M "cortex_m::quantized_add.out"
    NS "cortex_m_ns::quantized_mean.out"
    PORTABLE "aten::sub.out" "aten::clamp.out"
    MODULES haot_aaaaaaaaaa)
  write_sidecar(b PTE "b"
    CORTEX_M "cortex_m::quantized_linear.out" "cortex_m::quantized_add.out"
    PORTABLE "aten::clamp.out" "aten::leaky_relu.out"
    MODULES haot_bbbbbbbbbb haot_aaaaaaaaaa)
  nsx_executorch_configure_from_sidecars("${WORK}/a.pte.json" "${WORK}/b.pte.json")
  expect(union.cortex_m "${NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST}"
         "cortex_m::quantized_add.out,cortex_m::quantized_linear.out")
  expect(union.ns "${NSX_EXECUTORCH_NS_SELECT_OPS_LIST}" "cortex_m_ns::quantized_mean.out")
  expect(union.portable "${NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST}"
         "aten::clamp.out,aten::leaky_relu.out,aten::sub.out")
  expect(union.ns_ops "${NSX_EXECUTORCH_ENABLE_NS_OPS}" ON)
  expect(union.helia_aot "${NSX_EXECUTORCH_ENABLE_HELIA_AOT}" ON)
  expect(union.modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}" "haot_aaaaaaaaaa;haot_bbbbbbbbbb")
endfunction()
case_union()

# -- extra portable operators from -D are kept ---------------------------------
function(case_extra_portable)
  write_sidecar(c PTE "c" PORTABLE "aten::sub.out")
  set(NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST "aten::mean.out, aten::sub.out")
  nsx_executorch_configure_from_sidecars("${WORK}/c.pte.json")
  expect(extra.portable "${NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST}" "aten::mean.out,aten::sub.out")
  expect(extra.helia_aot "${NSX_EXECUTORCH_ENABLE_HELIA_AOT}" "")
endfunction()
case_extra_portable()

# -- extra cortex_m:: operators from -D; a NONE given there selects nothing ---
function(case_extra_cortex_m)
  write_sidecar(e PTE "e" CORTEX_M "cortex_m::quantized_add.out")
  set(NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST "cortex_m::softmax.out")
  set(NSX_EXECUTORCH_NS_SELECT_OPS_LIST "NONE")
  nsx_executorch_configure_from_sidecars("${WORK}/e.pte.json")
  expect(extra.cortex_m "${NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST}"
         "cortex_m::quantized_add.out,cortex_m::softmax.out")
  expect(extra.ns "${NSX_EXECUTORCH_NS_SELECT_OPS_LIST}" NONE)
  expect(extra.ns_ops "${NSX_EXECUTORCH_ENABLE_NS_OPS}" "")
endfunction()
case_extra_cortex_m()

# -- NSX_EXECUTORCH_REGISTER_ALL_OPS empties the cortex_m and NS lists, even ones
# -- an earlier configure cached, and switches the NS operators on with the ns provider
function(case_register_all)
  write_sidecar(f PTE "f" CORTEX_M "cortex_m::quantized_add.out" PORTABLE "aten::sub.out"
    MODULES haot_dddddddddd)
  set(NSX_EXECUTORCH_REGISTER_ALL_OPS ON)
  set(NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST "cortex_m::softmax.out")  # as cached by an earlier configure
  set(NSX_EXECUTORCH_NS_SELECT_OPS_LIST "NONE")
  nsx_executorch_configure_from_sidecars("${WORK}/f.pte.json")
  expect(all.cortex_m "${NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST}" "")
  expect(all.ns "${NSX_EXECUTORCH_NS_SELECT_OPS_LIST}" "")
  expect(all.ns_ops "${NSX_EXECUTORCH_ENABLE_NS_OPS}" ON)  # the delegate selected the ns provider
  expect(all.portable "${NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST}" "aten::sub.out")
  expect(all.helia_aot "${NSX_EXECUTORCH_ENABLE_HELIA_AOT}" ON)
endfunction()
case_register_all()
function(case_register_all_arm)
  write_sidecar(g PTE "g" CORTEX_M "cortex_m::quantized_add.out")
  set(NSX_EXECUTORCH_REGISTER_ALL_OPS ON)
  set(NSX_EXECUTORCH_CMSIS_NN_PROVIDER arm)
  nsx_executorch_configure_from_sidecars("${WORK}/g.pte.json")
  expect(all_arm.cortex_m "${NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST}" "")
  expect(all_arm.ns_ops "${NSX_EXECUTORCH_ENABLE_NS_OPS}" "")  # cortex_m_ns:: needs the ns provider
endfunction()
case_register_all_arm()

# -- an option already ON stays ON ---------------------------------------------
function(case_never_switched_off)
  write_sidecar(d PTE "d" CORTEX_M "cortex_m::quantized_add.out")
  set(NSX_EXECUTORCH_ENABLE_NS_OPS ON)
  nsx_executorch_configure_from_sidecars("${WORK}/d.pte.json")
  expect(on.ns_ops "${NSX_EXECUTORCH_ENABLE_NS_OPS}" ON)
  # No sidecar lists a cortex_m_ns:: operator, but NONE would cancel the option set by hand.
  expect(on.ns_list "${NSX_EXECUTORCH_NS_SELECT_OPS_LIST}" "")
endfunction()
case_never_switched_off()

# -- a sidecar from before the heliaAOT delegate (no helia_aot_modules key) ----
function(case_old_sidecar)
  write_sidecar(old PTE "old" CORTEX_M "cortex_m::quantized_add.out" NO_MODULES_KEY)
  nsx_executorch_configure_from_sidecars("${WORK}/old.pte.json")
  expect(old.helia_aot "${NSX_EXECUTORCH_ENABLE_HELIA_AOT}" "")
  expect(old.modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}" "")
endfunction()
case_old_sidecar()

# -- a sidecar without its PTE next to it is read without the SHA check -------
function(case_no_pte)
  write_sidecar(lonely SHA "0000" MODULES haot_cccccccccc)
  nsx_executorch_configure_from_sidecars("${WORK}/lonely.pte.json")
  expect(no_pte.modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}" haot_cccccccccc)
endfunction()
case_no_pte()

# -- one call per model adds to what the earlier calls set -----------------------
function(case_one_call_per_model)
  write_sidecar(first PTE "first" PORTABLE "aten::sub.out" MODULES haot_1111111111)
  write_sidecar(second PTE "second" PORTABLE "aten::clamp.out" MODULES haot_2222222222)
  nsx_executorch_configure_from_sidecars("${WORK}/first.pte.json")
  nsx_executorch_configure_from_sidecars("${WORK}/second.pte.json")
  expect(per_model.portable "${NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST}" "aten::clamp.out,aten::sub.out")
  expect(per_model.modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}" "haot_1111111111;haot_2222222222")
endfunction()
case_one_call_per_model()

# -- the helper included in one scope works when called from another -----------
function(_call_from_a_function sidecar)
  include("${_helper}")
  nsx_executorch_configure_from_sidecars("${sidecar}")
  set(NSX_EXECUTORCH_HELIA_AOT_MODULES "${NSX_EXECUTORCH_HELIA_AOT_MODULES}" PARENT_SCOPE)
endfunction()
function(case_other_scope)
  write_sidecar(scoped PTE "scoped" MODULES haot_3333333333)
  _call_from_a_function("${WORK}/scoped.pte.json")
  _call_from_a_function("${WORK}/scoped.pte.json")
  expect(scope.modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}" haot_3333333333)
endfunction()
case_other_scope()

# -- a model that needs the ns provider selects it, when none is chosen ---------
function(case_provider)
  write_sidecar(needs_ns PTE "needs ns" MODULES haot_4444444444)
  nsx_executorch_configure_from_sidecars("${WORK}/needs_ns.pte.json")
  expect(provider.selected "${NSX_EXECUTORCH_CMSIS_NN_PROVIDER}" ns)
endfunction()
case_provider()
function(case_provider_not_needed)
  write_sidecar(stock PTE "stock" CORTEX_M "cortex_m::quantized_add.out")
  nsx_executorch_configure_from_sidecars("${WORK}/stock.pte.json")
  expect(provider.untouched "${NSX_EXECUTORCH_CMSIS_NN_PROVIDER}" "")
endfunction()
case_provider_not_needed()

# -- more modules than the registry's default size raise it ----------------------
function(case_registry_size)
  write_sidecar(many PTE "many" MODULES m0 m1 m2 m3 m4 m5 m6 m7 m8)
  nsx_executorch_configure_from_sidecars("${WORK}/many.pte.json")
  expect(registry.raised "${NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS}" 9)
  write_sidecar(few PTE "few" MODULES m0 m1)
  set(NSX_EXECUTORCH_HELIA_AOT_MODULES "")
  set(NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS 12)
  nsx_executorch_configure_from_sidecars("${WORK}/few.pte.json")
  expect(registry.kept "${NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS}" 12)
endfunction()
case_registry_size()

# -- a sidecar under another name, next to a directory named like its "PTE" -----
function(case_renamed_sidecar)
  write_sidecar(renamed SHA "0000" MODULES haot_5555555555)
  file(RENAME "${WORK}/renamed.pte.json" "${WORK}/renamed.json")
  file(MAKE_DIRECTORY "${WORK}/renamed")
  nsx_executorch_configure_from_sidecars("${WORK}/renamed.json")
  expect(renamed.modules "${NSX_EXECUTORCH_HELIA_AOT_MODULES}" haot_5555555555)
endfunction()
case_renamed_sidecar()

# -- errors ---------------------------------------------------------------------
write_sidecar(stale PTE "stale program" SHA "0123")
expect_failure(stale_pte "describes a PTE with SHA-256 0123" "${WORK}/stale.pte.json")

write_sidecar(schema PTE "schema" SCHEMA "nsx-executorch.pte-manifest/99")
expect_failure(schema "is not a PTE sidecar this version reads" "${WORK}/schema.pte.json")

file(WRITE "${WORK}/broken.pte.json" "{ not json")
expect_failure(broken_json "is not a PTE sidecar this version reads" "${WORK}/broken.pte.json")

expect_failure(missing "does not exist" "${WORK}/missing.pte.json")

expect_failure(none "no sidecars given" "")
# A quoted variable that is unset is one empty argument, not no arguments.
expect_failure(none_quoted "no sidecars given" "" -DQUOTED=ON)

# A delegated sidecar whose module list cannot be read must not configure a
# firmware without the delegate.
foreach(_bad null "\"haot_x\"" "[]")
  write_sidecar(malformed PTE "malformed" MODULES haot_6666666666)
  file(READ "${WORK}/malformed.pte.json" _json)
  string(JSON _json SET "${_json}" helia_aot_modules "${_bad}")
  file(WRITE "${WORK}/malformed.pte.json" "${_json}")
  expect_failure("malformed_modules(${_bad})" "helia_aot_modules" "${WORK}/malformed.pte.json")
endforeach()

write_sidecar(wants_ns PTE "wants ns" NS "cortex_m_ns::quantized_mean.out")
expect_failure(provider "need NSX_EXECUTORCH_CMSIS_NN_PROVIDER=ns, but the provider is 'arm'"
  "${WORK}/wants_ns.pte.json" -DNSX_EXECUTORCH_CMSIS_NN_PROVIDER=arm)

get_property(_failures GLOBAL PROPERTY sidecar_test_failures)
list(LENGTH _failures _count)
if(_count GREATER 0)
  message(FATAL_ERROR "sidecar tests: ${_count} failed")
endif()
message(STATUS "sidecar tests: all passed")
