# ExecuTorch build options and compiler flags

Audit record for issue #5. Every stock ExecuTorch option this module forces,
every relevant option it deliberately leaves alone, and how compiler flags
reach the ExecuTorch subtree. Values were verified against ExecuTorch v1.3.0
(`external/executorch`, pinned `3a97429`) and against real firmware build
trees (`nsx build` app workspaces and the `tests/real-provider` harness), not
inferred from documentation.

## How flags reach ExecuTorch targets

`CMakeLists.txt` `add_subdirectory()`s stock ExecuTorch, then walks every
concrete target in that subtree and links one of two INTERFACE flag carriers
into it:

- `nsx_executorch_module_flags` — the board ABI flags plus this module's own
  optimization and hygiene flags. Applied to every target that has not chosen
  its own `-O` level.
- `nsx_executorch_abi_flags` — the board ABI flags only. Applied to targets
  that pin their own optimization (in practice the CMSIS-NN provider, which
  builds at `-Ofast` via `NSX_CMSIS_NN_OPTIMIZATION` / `CMSIS_OPTIMIZATION_LEVEL`).

Three ordering facts make this design necessary; they were the root causes of
the bugs this audit found:

1. `nsx::board_flags` carries the *app's* policy flags (`-g -O3 -ffast-math`
   for GCC/ATfE, `-Ofast` for armclang) in addition to the ABI flags
   (`nsx_toolchain_flags.cmake` in neuralspotx). Compile options land after
   `CMAKE_<LANG>_FLAGS_<CONFIG>` on the command line and the last `-O` wins,
   so copying the raw board interface onto ExecuTorch targets silently
   overrode every optimization choice the subtree made. Before this audit the
   entire runtime shipped at `-O3 -ffast-math` while
   `EXECUTORCH_OPTIMIZE_SIZE=ON` suggested `-Os`.
2. The CMSIS-NN provider links `nsx::board_flags` `PUBLIC`, so its dependents
   (`cortex_m_kernels`, `cortex_m_ns_kernels`) inherit the raw board options
   through `LINK_LIBRARIES` — *after* their own `COMPILE_OPTIONS`. The only
   position where our `-O` choice and `-fno-fast-math` reliably win is a flag
   carrier linked after that inheritance, which is why the walk links
   INTERFACE targets instead of calling `target_compile_options()`.
3. Stock ExecuTorch only applies `-Os`/`-fno-exceptions`/`-fno-rtti`/no-unwind
   under `CMAKE_BUILD_TYPE=Release` (it edits `CMAKE_*_FLAGS_RELEASE`; with an
   empty build type it silently defaults itself to Debug, i.e. `-O0`).
   `nsx configure` always passes `Release`, but standalone consumers may not —
   so the module applies those flags unconditionally through
   `nsx_executorch_module_flags` and does not depend on the build type.

`-ffast-math` is deliberately stripped from everything except the CMSIS-NN
provider: it changes float semantics (no NaN/Inf guarantees, unsafe
reassociation) inside runtime and kernel code that this module does not own
the numerics of. The provider keeps its own `-Ofast` — that is its published
performance contract, and the arm/ns providers are byte-identical in flags.

## Module knobs

| Knob | Default | Meaning |
| --- | --- | --- |
| `NSX_EXECUTORCH_OPTIMIZATION` | `speed` | `speed` = `-O3`, `size` = `-Os` for the ExecuTorch subtree, the codegen'd ops libs, and the adapter. `speed` preserves continuity with all previously published Tier-1 numbers (which were de-facto `-O3`). Hold it constant across arm-vs-ns comparisons. |
| `NSX_EXECUTORCH_ENABLE_PROGRAM_VERIFICATION` | `OFF` | Forwarded to `EXECUTORCH_ENABLE_PROGRAM_VERIFICATION`. See rationale below. |
| `NSX_EXECUTORCH_ENABLE_PROFILING` | `OFF` | Forwarded to `EXECUTORCH_BUILD_DEVTOOLS` + `EXECUTORCH_ENABLE_EVENT_TRACER`; adds `ET_EVENT_TRACER_ENABLED` to every consumer-scope target that compiles ExecuTorch headers (see "Definition consistency"). |

## Forced stock options, and why

| Option | Value | Why |
| --- | --- | --- |
| `EXECUTORCH_BUILD_EXECUTOR_RUNNER` | OFF | Host/posix demo runner; the NSX app is the runner. |
| `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER` | ON | `BufferDataLoader` used by the adapter. Pulls in nothing else (it is a dependee of other extensions, never a dependent). |
| `EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR` | OFF | Only needed for external `.ptd` tensor files; single-PTE named data is handled inside `executorch_core` by `internal::PteDataMap` (`program.cpp`). |
| `EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL` / `_RUNNER_UTIL` | OFF | Debug/print helpers for the demo runner. |
| `EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP` | OFF | Same rationale as FLAT_TENSOR: external-file case only. |
| `EXECUTORCH_BUILD_PORTABLE_OPS` | derived | ON iff `NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST` is non-empty. Note the full ~207-kernel `portable_kernels` glob is always compiled in that case; selection controls *registration*, and `--gc-sections` at the app link drops the unreferenced kernels. |
| `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` | OFF | Targets AArch64/x86 (sleef, OpenMP); useless on Cortex-M and incompatible with dtype-selective build. |
| `EXECUTORCH_BUILD_KERNELS_QUANTIZED` | OFF | Quantized ATen fallbacks; the cortex_m/CMSIS-NN path covers the quantized ops we ship. |
| `EXECUTORCH_BUILD_CORTEX_M` | ON | Builds `cortex_m_kernels` + `cortex_m_ops_lib` (16 CMSIS-NN-backed ops) — the reason this module exists. `CMSIS_NN_LOCAL_PATH` points at the provider adapter so the pinned NSX provider is the single CMSIS-NN source; `FETCHCONTENT_FULLY_DISCONNECTED=ON` turns any silent fallback fetch into a configure error. |
| `EXECUTORCH_BUILD_PTHREADPOOL` / `_CPUINFO` | OFF | Host-only dependencies; bare metal has neither threads nor cpuinfo. (These are the two options `EXECUTORCH_BUILD_ARM_BAREMETAL` would default OFF — forcing them is equivalent and explicit.) |
| `EXECUTORCH_BUILD_TESTS` | OFF | Also keeps googletest out (`BUILD_TESTING=OFF`) and avoids TESTS force-enabling FLAT_TENSOR. |
| `EXECUTORCH_BUILD_DEVTOOLS` | = profiling | Required by `EXECUTORCH_ENABLE_EVENT_TRACER`. Configures `etdump`/`bundled_program`/host-`flatcc`, but with `EXCLUDE_FROM_ALL` nothing builds unless linked — the adapter does not link etdump, so the flash cost of DEVTOOLS=ON is zero today. |
| `EXECUTORCH_ENABLE_EVENT_TRACER` | = profiling | Defines `ET_EVENT_TRACER_ENABLED`, a *header-inlined ABI switch* — see "Definition consistency". |
| `EXECUTORCH_ENABLE_LOGGING` | OFF | `ET_LOG()` collapses to `((void)0)`; verified no format strings survive in the archives. With the `minimal` PAL even enabled logging would emit nowhere; a strong `et_pal_emit_log_message` override is needed first. |
| `EXECUTORCH_ENABLE_PROGRAM_VERIFICATION` | = knob (OFF) | The verifier (~46 generated `Verify*` functions + `validate_program()`) runs only when `Program::load` is called with `Verification::InternalConsistency`; the adapter loads with the default `Minimal`, so the code was linked but never executed. NSX deployments SHA-check the PTE via the helia-torch sidecar manifest instead. Flash cost measured below. |
| `EXECUTORCH_OPTIMIZE_SIZE` | = knob | Kept consistent with `NSX_EXECUTORCH_OPTIMIZATION`, but the authoritative `-O` is the explicit per-target flag (Release-only variable edits do not survive the board-interface override, and do nothing under other build types). |
| `EXECUTORCH_USE_DL` | OFF | Only gates `find_library(dl)`; no `dl` on newlib. It never defines `ET_USE_LIBDL`, so ON would be a no-op-at-best. |
| `EXECUTORCH_PAL_DEFAULT` | `minimal` | posix PAL needs a hosted libc. Note `minimal` provides *weak* stubs: `et_pal_current_ticks` returns a sentinel (11223344) — profiling timestamps come from the app's strong override, and `et_pal_abort` is `__builtin_trap()`. |
| `EXECUTORCH_SELECT_OPS_LIST` | = portable list | Registration-selective build for portable fallbacks. Entries are fully-qualified overload names (`aten::mean.out`). The helia-torch sidecar computes this list per model. |
| `MAX_KERNEL_NUM` | 64 / 96 (ns ops) | Static registry: 12 B `.bss` per slot. Static-init registrations: 25 prim ops + 16 cortex_m (+5 cortex_m_ns), so 64 leaves 23 slots and 96 leaves 50 for the portable list. Overflow aborts **before `main()` and silently** (logging off + minimal PAL ⇒ `__builtin_trap`), so keep headroom visible. The pinned value also disables ExecuTorch's auto-sizing, which would undercount (it cannot see the cortex_m registrations). |

## Options evaluated and deliberately not used

| Option | Verdict |
| --- | --- |
| `EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD` | Unusable in v1.3.0. Its `REQUIRES EXECUTORCH_SELECT_OPS_MODEL` guard misfires on STRING options (`check_required_options_on` dereferences the value as a variable name), it only works with `SELECT_OPS_MODEL` (a `.pte`, not an ops list), and reading the PTE needs the compiled `executorch.codegen.tools.selective_build` pybind module, which the vendored torchgen-only host tooling cannot provide. Revisit on an ExecuTorch upgrade — with a portable fallback list in play it is the biggest remaining size lever. |
| `EXECUTORCH_SELECT_OPS_MODEL` / `_YAML` | Same pybind dependency; also the `.pte` path is not tracked as a build dependency (swapping the model does not re-run codegen), and the declared mutual-exclusion checks between the three SELECT_OPS options never fire. The hand-maintained (sidecar-computed) `SELECT_OPS_LIST` stays. |
| `EXECUTORCH_BUILD_ARM_BAREMETAL` | Ethos-U only. It builds `executorch_delegate_ethos_u`, which hard-links an `ethosu_core_driver` target nothing in this tree creates — enabling it breaks the configure. It sets no CPU/ABI flags and no generic Cortex-M behaviour. |
| `EXECUTORCH_ENABLE_BUNDLE_IO` | The define is consumed only by example runners in v1.3.0; no runtime-library code reads it. |
| `EXECUTORCH_FLATBUFFERS_MAX_ALIGNMENT` | Left at the ExecuTorch default (1024, vs upstream flatbuffers' 32). Zero runtime cost; must be ≥ the tensor-data alignment chosen at export. |
| `EXECUTORCH_BUILD_SHARED` | Would force `CMAKE_POSITION_INDEPENDENT_CODE=ON` over the module's explicit OFF. Static-only on bare metal. |
| `EXECUTORCH_LOG_LEVEL` | Moot while logging is OFF (`ET_MIN_LOG_LEVEL` is passed but unused in that configuration). |

## Definition consistency (consumer-scope targets)

Stock ExecuTorch applies its config defines via directory-scoped
`add_definitions()`, which do **not** reach targets created in this module's
scope even though they compile the same headers. `ET_EVENT_TRACER_ENABLED` in
particular gates inline code in `event_tracer_hooks.h` and `method.cpp`
paths; a TU compiled without it takes untraced code paths and can diverge
from `executorch_core`. The module therefore mirrors
`ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0`, `ET_MIN_LOG_LEVEL=Info`, and
(under profiling) `ET_EVENT_TRACER_ENABLED` onto `nsx_executorch`,
`cortex_m_ns_kernels`, and `cortex_m_ns_ops_lib`. (`ET_LOG_ENABLED=0`
propagates by itself: it is a PUBLIC define on the `executorch` targets.)

Before this audit `cortex_m_ns_kernels`/`cortex_m_ns_ops_lib` compiled
without `ET_EVENT_TRACER_ENABLED` in profiling firmware — confirmed in real
`hpx` build trees.

## Verified flag outcomes (tests/real-provider, GCC 15.2, Cortex-M55)

| Target group | Effective flags |
| --- | --- |
| ExecuTorch runtime, portable + cortex_m(+ns) kernels, codegen'd ops libs, etdump/flatccrt (if linked), adapter | `-mcpu=cortex-m55 -mthumb -mfloat-abi=hard -fshort-enums` + `-O3`/`-Os` (knob) + `-fno-fast-math -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti` + no unwind tables |
| CMSIS-NN provider | its own `-Ofast` (+ board policy flags; unchanged, provider-owned) |

No `-mfpu` is passed on Cortex-M55 (matching `nsx_apply_toolchain_flags`):
GCC then uses the full `fpv5-d16` + MVE the `-mcpu` implies, and clang keeps
`__ARM_FEATURE_MVE=3`. CMSIS-NN's MVE path (`ARM_MATH_MVEI`) derives from the
`__ARM_FEATURE_MVE` predefine — i.e. solely from `-mcpu=cortex-m55` — in both
providers; there is no CMake define to check, only the flag.

The harness now sets `CMAKE_BUILD_TYPE=Release` by default (mirroring
`nsx configure`) and carries the full production board interface including
`-g -O3 -ffast-math`, so the ABI-vs-policy filtering above is exercised by CI
exactly as in an app build.

## Measurements (Apollo510 EVB, GCC 15.2, 96 MHz LP clock)

Median of 25 iterations via `hpx profile`; flash = `text` from
`arm-none-eabi-size` on the profiler firmware (which includes the profiler
app and etdump, so deltas, not absolutes, are the meaningful numbers).

### tier1_arm (arm provider, 3 portable fallback ops)

| Configuration | text (B) | avg cycles | vs previous ship |
| --- | --- | --- | --- |
| Pre-audit (de-facto `-O3 -ffast-math`, verification ON) | 267,376 | 3,507,826 | — |
| **`speed` defaults** (`-O3 -fno-fast-math`, verification OFF) | 249,992 | 3,637,987 | **−6.5% flash, +3.7% cycles** |
| `speed` + verification ON | 262,944 | 3,635,204 | verifier = **+12,952 B**, latency unchanged |
| `size` (`-Os -fno-fast-math`, verification OFF) | 199,160 | 4,874,113 | −25.5% flash, +38.9% cycles |

The +3.7% on the `speed` default is **entirely** the portable
`aten::leaky_relu.out` fallback (43.7k → 172.8k cycles): without fast-math
GCC will not vectorize its float compare-select. Every other op is within
±2% (portable `clamp` improves 3.8%). The ns provider serves leaky_relu with
a native CMSIS-NN kernel and does not hit this path.

### tier1_ns (ns provider, ns ops, no portable fallbacks)

| Configuration | text (B) | avg cycles |
| --- | --- | --- |
| Pre-audit | 225,824 | 2,761,860 |
| **`speed` defaults** | 209,232 (−7.3%) | 2,762,249 (+0.01%) |

Where CMSIS-NN kernels serve every op, removing `-ffast-math` from the
runtime is latency-neutral, and the flash win is the verifier plus dead
unwind/RTTI machinery. Per-layer etdump tracing verified working with the
`ET_EVENT_TRACER_ENABLED` fix on the ns op library.

### Decisions taken from the data

- **Default `NSX_EXECUTORCH_OPTIMIZATION=speed`**: `-Os` costs +39% cycles
  on tier1_arm — unacceptable for published latency numbers. `size` remains
  a knob for flash-constrained deployments (−50.8 KB vs speed).
- **Default `-fno-fast-math` everywhere except the CMSIS-NN provider**:
  correct float semantics in code we don't own the numerics of, at zero cost
  on the ns path and +3.7% on tier1_arm concentrated in one fragile portable
  kernel. If that op matters in an arm-provider deployment, prefer the ns
  provider (native kernel) over re-enabling fast-math.
- **Default `NSX_EXECUTORCH_ENABLE_PROGRAM_VERIFICATION=OFF`**: 12,952 B of
  flash for a verifier the adapter never invokes; integrity is covered by
  the sidecar SHA-256. Latency confirmed unaffected either way.

