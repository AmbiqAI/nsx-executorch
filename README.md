# nsx-executorch

Thin NeuralSPOT-X adapter for the stock ExecuTorch Cortex-M runtime.

The package keeps ExecuTorch unmodified and supplies:

- an NSX CMake target, `nsx::executorch`;
- caller-owned, heap-free method and tensor arenas;
- total `Method::execute()` cycle measurement;
- optional per-operator callbacks backed by ExecuTorch `EventTracer`;
- deterministic, offline source and host-tool pins.

## Kernel providers

Set `NSX_EXECUTORCH_CMSIS_NN_PROVIDER` before adding the module:

```cmake
set(NSX_EXECUTORCH_CMSIS_NN_PROVIDER arm CACHE STRING "" FORCE)
```

- `arm` (default) resolves the `arm-cmsis-nn` NSX module (repo
  `AmbiqAI/arm-cmsis-nn`, exported target `nsx::arm_cmsis_nn`), which itself
  vendors pinned, stock upstream CMSIS-NN and CMSIS 6 sources.
- `ns` resolves `nsx-cmsis-nn` v7.29.2 (project repo
  `AmbiqAI/ns-cmsis-nn`). Its NSX target is bridged to the stock
  `cmsis-nn` target expected by ExecuTorch.

Neither CMSIS-NN nor CMSIS 6 is vendored directly in this repository anymore;
both are consumed as NSX modules. NSX does not resolve optional dependencies
automatically, so an app must list the selected provider immediately before
`nsx-executorch` in `nsx.yml`:

```yaml
modules:
  - name: arm-cmsis-nn  # or nsx-cmsis-nn
  - name: nsx-executorch
```

Set `NSX_EXECUTORCH_CMSIS_NN_PROVIDER` before `nsx_bootstrap_app()`. Keeping
the provider first lets NSX configure it exactly once despite stock
ExecuTorch's non-idempotent CMSIS-NN `add_subdirectory()`.

For standalone use (development, CI, helia-profiler materialization, or direct
`add_subdirectory()`), set the provider root in the CMake cache:

```cmake
# provider=arm
set(NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT "/path/to/arm-cmsis-nn" CACHE PATH "" FORCE)
# provider=ns
set(NSX_EXECUTORCH_NS_CMSIS_NN_ROOT "/path/to/ns-cmsis-nn" CACHE PATH "" FORCE)
```

The ns override accepts either the repository root or its `nsx/` module
directory. `NSX_EXECUTORCH_SOURCE_ROOT` similarly overrides the pinned
ExecuTorch checkout. CMake never fetches dependencies: all provider and
ExecuTorch sources must be materialized before configure. This is a
build-tree NSX module; stock ExecuTorch package-install/export rules are
suppressed because they cannot encode an app-selected NSX provider.

Both providers consume the same stock Cortex-M operators and the same PTE.
Provider selection never changes operator schemas or ExecuTorch lowering.
NS-CMSIS-NN v7.29.2 adds a weight-sum context to convolution, depthwise
convolution, and transpose-convolution wrappers. A private adapter extends
ExecuTorch's temporary allocations, precomputes those sums, and passes the
extra contexts without modifying the pinned ExecuTorch source.

Enable layer callbacks with:

```cmake
set(NSX_EXECUTORCH_ENABLE_PROFILING ON CACHE BOOL "" FORCE)
```

Each callback receives an `OperatorEvent`: the ExecuTorch chain and
instruction identifiers plus what the instruction runs — the serialized
operator name and overload (`aten::add` / `out`,
`cortex_m::quantized_conv2d` / `out`) for kernel calls, the backend id for
delegate calls. A caller such as Helia Profiler labels its per-layer rows from
the name and samples PMU counters around every operator.

## NS additional kernels (`cortex_m_ns::`)

The ns provider ships kernels beyond stock CMSIS-NN. This repo can expose a
first tier of them — `sub`, `hardswish`, `mean`, standalone
`relu`/`relu6`/`hardtanh`/`clamp`, and `leaky_relu` — to ExecuTorch models as
quantized int8 operators in a dedicated `cortex_m_ns::` namespace. Everything
composes stock ExecuTorch through public extension points (`ops-ns/` operator
YAML + codegen, `aot/nsx_cortex_m/` Python package); the pinned submodule is
never modified.

Namespace contract:

- PTEs containing only stock `cortex_m::` ops run on **both** providers.
- PTEs containing `cortex_m_ns::` ops require an ns build with NS ops
  enabled; on any other build they fail fast at `Method::load()` with
  "operator missing" instead of silently miscomputing.

Runtime gating (default OFF; provider=arm builds are byte-identical and gain
no new targets):

```cmake
set(NSX_EXECUTORCH_CMSIS_NN_PROVIDER ns CACHE STRING "" FORCE)
set(NSX_EXECUTORCH_ENABLE_NS_OPS ON CACHE BOOL "" FORCE)
```

| provider | `NSX_EXECUTORCH_ENABLE_NS_OPS` | result |
|----------|--------------------------------|--------|
| arm      | OFF (default)                  | stock build, no new targets |
| arm      | ON                             | configure error (fail fast) |
| ns       | OFF (default)                  | stock ns build, no new targets |
| ns       | ON                             | adds `cortex_m_ns_kernels` + `cortex_m_ns_ops_lib`, raises `MAX_KERNEL_NUM` to 96 |

Enabling NS ops additionally requires PyYAML on the host Python (ExecuTorch
codegen); the vendored `tools/python` tree covers torchgen only.

AOT export (requires a host ExecuTorch install, see `aot/`):

```python
from nsx_cortex_m import export

result = export(model, example_inputs, kernel_provider="ns")  # or "arm"
result.write_pte("model.pte")           # also writes the model.pte.json sidecar
print(result.edge_ops)                  # lowered operator histogram
print(result.portable_select_ops_list)  # NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST value
```

## helia-torch CLI and PTE sidecars

`aot/` builds as the `helia-torch` pip package (see `aot/pyproject.toml`;
its torch/executorch pins must match this repository's pinned submodule).
The CLI is file-in/file-out: the user runs `torch.export.save()` in their
own training environment and hands the `.pt2` artifact to the tool — no
model code executes here.

```sh
helia-torch compile model.pt2 --provider ns --calibrate calib.npz
helia-torch report  model.pte    # kernel lowering report from the sidecar
helia-torch inspect model.pte    # sidecar manifest as JSON
```

`compile` writes `model.pte` plus a `model.pte.json` sidecar (schema
`nsx-executorch.pte-manifest/1`), SHA-256-bound to that exact PTE. The
sidecar records everything a target build needs to register the right
kernels and size its memory: kernel provider, `requires_ns_ops`, the
portable select-ops list, the planned arena size, and per-tensor I/O sizes.
Consumers verify the hash and self-configure — helia-profiler reads it
automatically, so a profiling config reduces to the model path, this
checkout's path, and a board. The compile step also prints a lowering
report that flags any `[PORTABLE]` fallback loudly: a portable op runs on
the standard float kernels and typically dominates per-op cost on target.

### Configuring firmware from sidecars

An app's CMake can take its nsx-executorch configuration from the sidecars
instead of copying operator lists and module names by hand
(`cmake/nsx_executorch_sidecar.cmake`):

```cmake
include(<nsx-executorch>/cmake/nsx_executorch_sidecar.cmake)

# Before nsx-executorch is configured (before nsx_bootstrap_app()):
nsx_executorch_configure_from_sidecars(build/kws.pte.json build/vww.pte.json)

# After it, once the app target exists:
nsx_executorch_link_helia_aot_modules(app
    MODULES_DIR build/helia_aot)    # omit when nsx.yml lists the modules
```

- `nsx_executorch_configure_from_sidecars()` sets the `cortex_m::`,
  `cortex_m_ns::` and portable select lists to the union of the operators the
  models still run in ExecuTorch (`NONE` for a namespace none of them uses;
  see "Selective operator registration"), switches on `NSX_EXECUTORCH_ENABLE_NS_OPS` and
  `NSX_EXECUTORCH_ENABLE_HELIA_AOT` when a model needs them, and records the
  heliaAOT modules the programs call. It sets normal variables, so they take
  precedence over the cache and follow the sidecars on every configure; a
  list given with `-D` is added to the union, and an option already ON stays
  ON.
- A model that needs `cortex_m_ns::` operators or heliaAOT needs the ns
  provider: the function selects `NSX_EXECUTORCH_CMSIS_NN_PROVIDER=ns` when
  none is set, and fails, naming the sidecar, when another one is. It also
  raises `NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS` when the models call more
  modules than the registry holds.
- Several sidecars configure what all of them need, so firmware can swap
  between a fixed set of models. Calling the function once per model does
  the same: each call adds to what the earlier ones set. Firmware that loads arbitrary models sets
  `NSX_EXECUTORCH_REGISTER_ALL_OPS=ON` (a cache option, so `-D` works), which
  registers every `cortex_m::` and, with the ns provider, `cortex_m_ns::`
  operator again, whatever lists an earlier configure cached.
- When the PTE sits next to its sidecar, its SHA-256 must match the one the
  sidecar records, so a stale sidecar fails the configure instead of
  `Method::load` on the target. Each sidecar, and its PTE when it is there,
  is a configure dependency: re-exporting a model reconfigures the build.
- `nsx_executorch_link_helia_aot_modules()` links each recorded heliaAOT
  module into the app, adding it from `MODULES_DIR` when it is not already in
  the build, and fails the configure when a module is in neither place.
  `MODULES <name>...` links those modules instead of the sidecars'; `MODULES`
  with no names links none.

helia-profiler reads the sidecar itself and does not need these functions.

### Selective operator registration

ExecuTorch registers kernels in a static table at init, so every registered
operator, and the CMSIS-NN kernels it calls, is linked whether or not the
model uses it. By default this module registers all 16 stock `cortex_m::`
operators (and all `cortex_m_ns::` operators with
`NSX_EXECUTORCH_ENABLE_NS_OPS`). For a deployment build, register just what
the model uses, which the sidecar already lists.
`nsx_executorch_configure_from_sidecars()` sets these lists from the sidecars
(see "Configuring firmware from sidecars"); to set them by hand:

```cmake
# From <model>.pte.json: operators.cortex_m and operators.cortex_m_ns.
# Normal variables, set before nsx-executorch is configured: they follow every
# edit made here (a cache entry would keep its first value). The option is
# the app's own, so -DAPP_REGISTER_ALL_OPS=ON still registers everything.
if(NOT APP_REGISTER_ALL_OPS)
  set(NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST
      "cortex_m::quantize_per_tensor.out,cortex_m::dequantize_per_tensor.out,cortex_m::quantized_linear.out")
  set(NSX_EXECUTORCH_NS_SELECT_OPS_LIST "cortex_m_ns::quantized_mean.out")
endif()
```

`NONE` registers no operators of that namespace and links no registration
library for it, for example when a PTE is only a heliaAOT delegate call. An
empty value still registers all.

With a list, the module generates its own registration library for exactly
those operators over the stock `cortex_m_kernels` (the pinned tree is
untouched) and links it instead of stock `cortex_m_ops_lib`; unregistered
kernels are never referenced and stay out of the image. Names are checked
against the operator schemas at configure time. As with
`NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST`, a PTE that needs an operator
outside the list fails at `Method::load` with a missing-operator error.

For firmware that loads arbitrary or swappable models, leave the lists empty:
that registers every operator, as without this option. For a fixed set of
models, join their sidecar lists instead (the sidecar functions do this; with
them, `NSX_EXECUTORCH_REGISTER_ALL_OPS=ON` registers everything). A list set by
hand is a CMake cache value, so it persists in a build directory; to return to
registering everything, configure with an empty value
(`-DNSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST= -DNSX_EXECUTORCH_NS_SELECT_OPS_LIST=`)
or use a fresh build directory.

Measured on an Apollo510 EVB with an MLPerf Tiny KWS DS-CNN, registering
only the model's 7 operators: `.text` 224,564 to 188,604 bytes (-35.1 KiB,
-16%), cycles unchanged. That build also had 3x3 depthwise routing and two
`cortex_m_ns::` conv operators with export-time weight sums, both since
removed, so a current build's figure differs somewhat.

Memory *placement* is intentionally not in the sidecar: sizes are export
facts, but which RAM region each caller-owned arena lives in is a
board-pressure decision made by the consumer (helia-profiler's
`engine.config.*_location` keys, or the application's linker attributes).
Splitting activations across multiple planned arenas at export time
(per-tensor placement) is tracked in
[issue #3](https://github.com/AmbiqAI/nsx-executorch/issues/3).

`kernel_provider="arm"` reproduces the stock flow exactly (zero
`cortex_m_ns::` ops). With `"ns"`, ops that fail a lowering qualifier stay
portable ATen ops (correct but slow) and are reported in
`portable_fallback_ops` so the matching
`NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST` value can be applied.

Tier 1 restrictions (documented qualifiers; anything else falls back):

- int8 per-tensor quantization only (int16 is out of scope for this tier).
- `sub`: channel broadcast only, mirroring stock `quantized_mul`, and only
  for channels_last tensors (stock `CortexMAddMulCheck` policy).
- `mean`: rank ≤ 4, contiguous (non-channels_last) tensors, compile-time
  reduce dims (maps onto `arm_mean_s8`'s NHWC axis mask; the element count is
  folded into the output requantization AOT).
- `relu`/`relu6`/`hardtanh`/`clamp`: compile-time scalar bounds; bounds are
  quantized into the output domain and requantization is supported.
- `leaky_relu`: compile-time positive slope (two AOT requant pairs).
- `hardswish`: uses `arm_hard_swish_precise_s8` with `prescale=0` (matches
  the pinned ns-cmsis-nn unit tests; guarded by `relu_q6 ≤ 2^22`).

Follow-up tiers (later PRs): sigmoid/tanh int16, data-movement ops, further
reductions, LSTM/SVDF.

## heliaAOT delegate (`HeliaAotBackend`)

helia-torch can hand the parts of a model that [heliaAOT](https://ambiqai.github.io/helia-aot/) compiles to
an ExecuTorch delegate. heliaAOT turns them into plain C with per-layer kernel selection and whole-graph
memory planning, so they run without the ExecuTorch interpreter's per-operator overhead. Everything else
runs as usual.

```python
from nsx_cortex_m.export import export
from nsx_cortex_m.helia_aot import HeliaAotOptions

result = export(model, example_inputs, kernel_provider="ns",
                helia_aot=HeliaAotOptions(output_dir="build/helia_aot"))  # pip install 'helia-torch[helia-aot]'
result.write_pte("model.pte")
```

**What gets delegated.** The partitioner runs on the quantized edge program before the NS passes, where each
int8 operator is still `dequantize -> aten op -> quantize`. heliaAOT's PT2E front end
(`helia_aot.converters.pt2e`, heliaAOT ≥ 1.5) decides support group by group and converts the partitions, so the
partitioner claims exactly what converts. In int8, that covers:

- convolution, depthwise and transposed convolution;
- linear and bmm;
- add, sub, mul, maximum and minimum;
- relu (and relu6 / ±1 / 0..1 bounds), leaky relu, hard swish and softmax;
- max and average pooling, mean, pad, transpose and reshape;
- requantization.

Float↔int8 conversions at a partition's edges stay ExecuTorch `cortex_m` kernels, so each delegate exchanges
int8 tensors. Anything else (for example a fused `hardtanh` whose bounds clamp inside the output's int8 range)
stays in ExecuTorch and lowers through the NS passes as usual.

**What export writes.**

- **The quantized program**, saved as `output_dir/haot_<digest>.pt2`. heliaAOT's plan for each partition names
  the model by this file (its name and the SHA-256 of its bytes, which `shasum -a 256` reproduces) and the
  partition's index, so a plan made for a partition matches the same file again. The file is the program without
  its debug metadata (stack traces and module paths, which PyTorch writes differently on every export), so
  exporting the same quantized model again gives the same file, digest and module names; a model that differs in
  any weight or operator gets others. It loads with `torch.export.load()`, and `helia-torch compile` rebuilds from
  it. `HeliaAotOptions(debug_program=True)` also saves `haot_<digest>.debug.pt2`, the program with that metadata.
- **One heliaAOT NSX module per partition** in `output_dir`, named `haot_<digest>_<partition>` after that file.
  Each module includes `et/<name>_et_registration.c`, which registers the module with the runtime.
- **The PTE.** It holds only the delegate blob (module name and int8 tensor contract), not the compiled code.
  For the MLPerf Tiny KWS DS-CNN the PTE shrinks to 9.4 KB.
- **The sidecar**, which records `requires_helia_aot` and `helia_aot_modules`.

**Firmware.** Build with `NSX_EXECUTORCH_CMSIS_NN_PROVIDER=ns` and `NSX_EXECUTORCH_ENABLE_HELIA_AOT=ON` (the arm
provider is rejected, because heliaAOT modules link ns-cmsis-nn). List the generated modules after
`nsx-cmsis-nn` in `nsx.yml`, and link them into the app. `nsx_executorch_configure_from_sidecars()` and
`nsx_executorch_link_helia_aot_modules()` do both from the sidecar (see "Configuring firmware from sidecars").

- `run_once()` registers `HeliaAotBackend` (`src/helia_aot_backend.cpp`).
- Each module's registration file is an INTERFACE source, so it is compiled into the application and always
  runs.
- At Method load the backend looks the module up by name (`includes-api/nsx_helia_aot.h`) and checks the
  blob's tensor contract.
- `execute()` copies the delegate's int8 inputs in, runs the module, and copies its outputs out.

**Numerics.** A delegate computes exactly what the non-delegated program computes:

- **Host tests.** `aot/tests/test_helia_aot.py` compiles each operator's generated module with ns-cmsis-nn on
  the host and compares it byte for byte with ExecuTorch's reference kernels (34 cases, including broadcasting operands, padding that is neither SAME nor VALID, and the whole KWS DS-CNN).
- **heliaAOT settings.** The delegate sets the heliaAOT attributes this needs. For example, `HARD_SWISH` runs
  the precise backend, like `arm_hard_swish_precise_s8`, not heliaAOT's default lookup table.
- **A reference that does not match its kernel.** ExecuTorch's Python reference for
  `cortex_m::quantized_avg_pool2d` averages in float and rounds half to even, but the kernel it describes,
  `arm_avgpool_s8`, rounds half away from zero. The tests hold the delegate to the kernel's behavior.
- **Operators ExecuTorch leaves in float.** nsx-executorch lowers a mean to `cortex_m_ns::quantized_mean` only
  for a contiguous tensor, so without the delegate a mean over a channels_last tensor (global average pooling)
  runs on the float kernels. The delegate computes it with the int8 kernel, which matches the float result to
  one LSB; the tests hold it to the int8 operator's reference.
- **On the board.** On an Apollo510 EVB (Cortex-M55 MVE, ns-cmsis-nn 7.47.0), the KWS DS-CNN through the
  ExecuTorch runtime and the delegate matched the non-delegated reference bit for bit on 8/8 vectors.
  `Method::execute()` took about 1.625 M cycles, against 1.653 M for the same model on the ExecuTorch Cortex-M
  kernels.

**Limits.**

- **One instance per module.** Each module has one static context, so a model cannot run twice concurrently.
- **Eight modules by default.** The registry holds `NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS` modules (default 8),
  one per delegated partition of every PTE the firmware runs. `helia-torch report` says when a model needs more,
  and the backend logs how many modules did not fit when a PTE names one it cannot find.
- **Module names** are at most 63 bytes; a longer `module_prefix` fails the export.
- **Delegates are int8 in, int8 out.** `export(helia_aot=...)` leaves the float conversions in ExecuTorch;
  lowering a float-boundary program to `HeliaAotBackend` directly is refused.
- **Transposed convolution** needs `output_padding == 0`, `groups == 1` and no dilation.
- **Profiling.** A delegate reports as one operator event (`HeliaAotBackend`).
- **heliaAOT version.** The generated modules need the heliaAOT version helia-torch was installed with
  (`helia-aot>=1.5,<2`).

## Source setup and validation

Clone with the pinned ExecuTorch submodule:

```sh
git clone --recurse-submodules https://github.com/AmbiqAI/nsx-executorch.git
```

For a minimal Cortex-M checkout, initialize `external/executorch` and the
ExecuTorch submodules used by its stock CMake configuration:

```sh
git submodule update --init external/executorch
git -C external/executorch submodule update --init \
  backends/xnnpack/third-party/FXdiv \
  third-party/flatbuffers third-party/flatcc third-party/gflags third-party/json
```

Then verify the exact revision and configure the profiling-enabled smoke
projects for both providers (using the checked-in test fixtures that stand in
for real `arm-cmsis-nn` / `ns-cmsis-nn` checkouts):

```sh
python3 tests/verify_source_pins.py
cmake -S tests/smoke -B build/configure-smoke-arm -G Ninja
cmake -S tests/smoke-ns -B build/configure-smoke-ns -G Ninja
cmake -S tests/smoke-standalone -B build/configure-standalone-arm -G Ninja \
  -DNSX_EXECUTORCH_TEST_PROVIDER=arm
cmake -S tests/smoke-standalone -B build/configure-standalone-ns -G Ninja \
  -DNSX_EXECUTORCH_TEST_PROVIDER=ns
```

CI additionally builds the stock Cortex-M kernels with the exact Arm and NS
provider revisions listed in `PROVENANCE.md`.

## Build options

`docs/build-options.md` is the audit record for every ExecuTorch option this
module forces (and the ones it deliberately leaves off), how compiler flags
reach the subtree, and hardware before/after numbers for the choices that
were changed. The short version: the module owns the subtree's optimization
level via `NSX_EXECUTORCH_OPTIMIZATION` (`speed` = `-O3` default, `size` =
`-Os`), strips the app policy flags (`-O3 -ffast-math`) that the board
interface would otherwise impose, and defaults
`NSX_EXECUTORCH_ENABLE_PROGRAM_VERIFICATION=OFF` (the sidecar SHA covers PTE
integrity). Hold `NSX_EXECUTORCH_OPTIMIZATION` constant across arm-vs-ns
comparisons.

## Toolchains

The module builds under `arm-none-eabi-gcc` and ATfE (Arm Toolchain for
Embedded, LLVM-based). `armclang` is not yet supported. Cross-compiling test
fixtures carry one toolchain file each: `tests/real-provider/arm-none-eabi-gcc.cmake`
and `tests/real-provider/atfe.cmake` (requires the `ATFE_ROOT` env var pointing
at an ATfE install with the newlib overlay; skip the ATfE legs when unset).
All per-toolchain accommodations in the module live in the single
`CMAKE_CXX_COMPILER_ID STREQUAL "Clang"` block at the end of `CMakeLists.txt`.

Validated: ATfE 22.1.0 (newlib overlay) on Cortex-M55 / Apollo510 EVB, both
providers, `NSX_EXECUTORCH_ENABLE_NS_OPS` on and off, plus portable ops via
`NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST=aten::clamp.out,aten::leaky_relu.out,aten::sub.out`
(the tier-1 arm PTE's fallback list). Hardware numbers for the tier-1 arm PTE (96 MHz LP,
median of 25, helia-profiler `hpx compare`): total 3.51M cycles (GCC 15.2)
vs 2.77M cycles (ATfE 22.1.0), −21%; firmware `.text` −5.5%. Most of the
delta is clang auto-vectorizing float portable kernels with MVE tail
predication (`aten::clamp.out` 63x); `aten::leaky_relu.out` regressed ~54%
under ATfE — see the issue #5 flag audit. Numerical parity between the
toolchains is not yet exercised: `hpx validate` still gates ExecuTorch to
GCC (AmbiqAI/helia-profiler#141).

Known toolchain differences (issue #6):

- The fixture board flags pass no `-mfpu` for Cortex-M55 (matching NSX's
  `nsx_apply_toolchain_flags`): an explicit `-mfpu=fpv5-sp-d16` downgrades
  clang to integer-only MVE (`__ARM_FEATURE_MVE=1`), silently breaking
  ExecuTorch's Helium float kernels; GCC ignores `-mfpu` for MVE derivation.
- Upstream CMSIS-NN defaults `CMSIS_OPTIMIZATION_LEVEL` to `-Ofast`, which
  clang 22 deprecates (warning only).
