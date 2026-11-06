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

With NS-CMSIS-NN >= 7.37.0 on MVEI builds, the adapter also routes s8 3x3
depthwise layers to the direct entries `arm_depthwise_conv_s8_opt_3x3_c64_s1`
(64 channels, stride 1) and `arm_depthwise_conv_s8_opt_3x3`, the same
selection heliaRT makes, before falling back to the generic wrapper. The
entries apply their own gate (3x3, `ch_mult` 1, 16-2048 channels, stride 1
or 2, padding 0 or 1, ...) and decline other layers without writing output, so the
wrapper handles everything else. Output is identical to the wrapper; scratch
for 3x3 candidates grows to the entries' size (3008 + input W x C + 16 bytes).
Older providers, the `arm` provider and non-MVE builds are unchanged. Define
`NSX_EXECUTORCH_DW_S8_OPT_3X3=0` to force the wrapper for A/B comparisons.

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
quantized int8 operators in a dedicated `cortex_m_ns::` namespace, lowers
convolutions with their weight sums precomputed, and runs layer norm on
heliaCORE's `arm_layer_norm` kernels (both below). Everything
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

### Selective operator registration

ExecuTorch registers kernels in a static table at init, so every registered
operator, and the CMSIS-NN kernels it calls, is linked whether or not the
model uses it. By default this module registers all 16 stock `cortex_m::`
operators (and all `cortex_m_ns::` operators with
`NSX_EXECUTORCH_ENABLE_NS_OPS`). For a deployment build, register just what
the model uses, which the sidecar already lists:

```cmake
# From <model>.pte.json: operators.cortex_m and operators.cortex_m_ns
set(NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST
    "cortex_m::quantize_per_tensor.out,cortex_m::dequantize_per_tensor.out,cortex_m::quantized_linear.out"
    CACHE STRING "" FORCE)
set(NSX_EXECUTORCH_NS_SELECT_OPS_LIST
    "cortex_m_ns::quantized_conv2d_ws.out" CACHE STRING "" FORCE)
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
outside the list fails at `Method::load` with a missing-operator error, so
keep the lists empty (register all) for firmware that loads arbitrary models.

Measured on an Apollo510 EVB with an MLPerf Tiny KWS DS-CNN (5 `cortex_m::`
and 2 `cortex_m_ns::` operators): `.text` 224,564 to 188,604 bytes
(-35.1 KiB, -16%), cycles unchanged.

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

### Export-time weight sums (`quantized_conv2d_ws`, `quantized_depthwise_conv2d_ws`)

ns-cmsis-nn's MVE convolution kernels read a per-output-channel weight-sum
buffer, `input_offset * sum(weights of channel j) + bias[j]` (int32,
wrapping). The stock `cortex_m::quantized_conv2d` /
`quantized_depthwise_conv2d` reach ns-cmsis-nn through the private adapter,
which has no per-operator state and recomputes that buffer on every
inference; heliaRT computes it once at Prepare. With the ns profile,
`NsConvWeightSumPass` (after the stock `ConvertToCortexMPass`) computes it
once at export and rewrites every conv and depthwise conv with constant
weights to the matching `cortex_m_ns::*_ws` operator, which takes the sums as
a constant int32 `[out_ch]` tensor. ns-cmsis-nn documents the buffer as read
only and reusable while filter, bias and input offset are unchanged, which
holds for constants.

- Numerics are unchanged: the operators run the same kernels (including the
  3x3 depthwise routing above) with the same buffer contents.
- `weight_sum_format` (currently 1) versions the layout. The runtime rejects
  other values at execution, and the sidecar records it as
  `weight_sum_format`, so a provider that changes the layout fails loudly
  instead of computing wrong outputs. Bump `WEIGHT_SUM_FORMAT`
  (`aot/nsx_cortex_m/weight_sums_ns.py`) and `kWeightSumFormat`
  (`ops-ns/op_quantized_conv2d_ws.cpp`) together.
- The PTE grows by 4 bytes per output channel per layer; on builds without
  MVE the kernels take no weight sums and the tensor is ignored.
- Like every `cortex_m_ns::` operator, such PTEs need a runtime built with
  `NSX_EXECUTORCH_CMSIS_NN_PROVIDER=ns` and `NSX_EXECUTORCH_ENABLE_NS_OPS=ON`.

Measured on an Apollo510 EVB (Cortex-M55 at 96 MHz, ns-cmsis-nn 7.47.0)
with an MLPerf Tiny KWS DS-CNN exported through helia-torch, same weights
with and without the pass: 1,675,317 to 1,653,132 cycles per inference
(-1.3%), about -2.2k cycles per 3x3 depthwise layer and -1.7k per 1x1 conv.

Follow-up tiers (later PRs): sigmoid/tanh int16, data-movement ops, further
reductions, LSTM/SVDF.

### Layer norm (`quantized_layer_norm`, `layer_norm`)

`NsLayerNormPass` lowers `torch.nn.LayerNorm` / `F.layer_norm` over the last dimension to ns-cmsis-nn's
`arm_layer_norm_s8` / `_s16` / `_f32` (ns-cmsis-nn branch `feat/layer-norm`). The stock flow runs a layer norm in
float, between dequantize and quantize, on the portable `aten.native_layer_norm` kernel.

- **Quantized.** Between quantized neighbours the layer norm becomes `cortex_m_ns::quantized_layer_norm` on the
  int8 (or int16) tensor. gamma and beta are folded with the input and output quantization into int32 constants at
  export (`aot/nsx_cortex_m/layer_norm_ns.py`). The kernel is exact against float layer norm for int8 outputs, so
  the numerics are the stock flow's.
- **Float.** A layer norm left in float becomes `cortex_m_ns::layer_norm`. It uses `arm_layer_norm_f32` when
  ns-cmsis-nn is built with `ARM_NN_ENABLE_F32`, and an equivalent C loop otherwise.
- **Not lowered:** a layer norm over more than the last dimension, over a non-contiguous (e.g. channels-last)
  tensor, or whose mean/rstd outputs are used, stays as it was.
- **Runtime.** `quantized_layer_norm` needs an ns-cmsis-nn with the layer-norm kernels; against an older one it
  builds but fails at runtime with `NotSupported`.

The integer algorithm is ns-cmsis-nn's (`Tests/UnitTest/layer_norm_reference.py`); `aot/tests/test_layer_norm.py`
checks the export parameters and the reference operator against it and against the C kernel itself.

On an Apollo510 EVB, a transformer-style block (49 tokens, Linear 64→128, LayerNorm, ReLU, Linear 128→64,
LayerNorm) through the ExecuTorch runtime took 420,871 cycles, against 679,020 with the stock lowering, and its
output matched the host bit for bit (8/8 vectors). The kernel alone runs at about 5–8 cycles per element.

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

**What gets delegated.** The partitioner runs on the program after the NS passes up to the stock
`ConvertToCortexMPass`, and asks the AIR frontend (`aot/nsx_cortex_m/helia_aot/frontend.py`) about each node,
so it claims exactly what converts. That covers all 16 `cortex_m::` and 5 `cortex_m_ns::` operators in int8:

- convolution, depthwise and transposed convolution;
- linear and bmm;
- add, sub, mul, maximum and minimum;
- relu (and relu6 / ±1 / 0..1 bounds), leaky relu, hard swish and softmax;
- max and average pooling, mean, pad, transpose and reshape;
- requantization.

Float↔int8 conversions at a partition's edges stay ExecuTorch `cortex_m` kernels, so each delegate exchanges
int8 tensors. Anything else (for example a float `sigmoid`, which has neither an AIR mapping nor a `cortex_m`
kernel) stays in ExecuTorch.

**What export writes.**

- **One heliaAOT NSX module per partition** in `output_dir`, named `haot_<hash>` (stable for the same AIR).
  Each module includes `et/<name>_et_registration.c`, which registers the module with the runtime.
- **The PTE.** It holds only the delegate blob (module name and int8 tensor contract), not the compiled code.
  For the MLPerf Tiny KWS DS-CNN the PTE shrinks to 9.4 KB.
- **The sidecar**, which records `requires_helia_aot` and `helia_aot_modules`.

**Firmware.** Build with `NSX_EXECUTORCH_CMSIS_NN_PROVIDER=ns` and `NSX_EXECUTORCH_ENABLE_HELIA_AOT=ON` (the arm
provider is rejected, because heliaAOT modules link ns-cmsis-nn). List the generated modules after
`nsx-cmsis-nn` in `nsx.yml`, and link them into the app.

- `run_once()` registers `HeliaAotBackend` (`src/helia_aot_backend.cpp`).
- Each module's registration file is an INTERFACE source, so it is compiled into the application and always
  runs.
- At Method load the backend looks the module up by name (`includes-api/nsx_helia_aot.h`) and checks the
  blob's tensor contract.
- `execute()` copies the delegate's int8 inputs in, runs the module, and copies its outputs out.

**Numerics.** A delegate computes exactly what the non-delegated program computes:

- **Host tests.** `aot/tests/test_helia_aot.py` compiles each operator's generated module with ns-cmsis-nn on
  the host and compares it byte for byte with ExecuTorch's reference kernels (26 cases, including the whole KWS DS-CNN).
- **heliaAOT settings.** The delegate sets the heliaAOT attributes this needs. For example, `HARD_SWISH` runs
  the precise backend, like `arm_hard_swish_precise_s8`, not heliaAOT's default lookup table.
- **A reference that does not match its kernel.** ExecuTorch's Python reference for
  `cortex_m::quantized_avg_pool2d` averages in float and rounds half to even, but the kernel it describes,
  `arm_avgpool_s8`, rounds half away from zero. The tests hold the delegate to the kernel's behavior.
- **On the board.** On an Apollo510 EVB (Cortex-M55 MVE, ns-cmsis-nn 7.47.0), the KWS DS-CNN through the
  ExecuTorch runtime and the delegate matched the non-delegated reference bit for bit on 8/8 vectors.
  `Method::execute()` took about 1.625 M cycles, against 1.653 M for the same model on the ExecuTorch Cortex-M
  kernels.

**Limits.**

- **One instance per module.** Each module has one static context, so a model cannot run twice concurrently.
- **Transposed convolution** needs `output_padding == 0` and `groups == 1`.
- **Profiling.** A delegate reports as one operator event (`HeliaAotBackend`).
- **heliaAOT version.** The generated modules need the heliaAOT version helia-torch was installed with
  (`helia-aot>=1.3,<2`).

### Whole model in heliaAOT, without ExecuTorch (`compile_model`)

When the partitioner claims every operator, the PTE is only `quantize -> HeliaAotBackend -> dequantize`: the
ExecuTorch runtime loads the program and calls the generated code. `compile_model()` produces that generated
code on its own, as one plain heliaAOT NSX module, with no PTE, ExecuTorch runtime or delegate registry:

```python
from nsx_cortex_m.helia_aot import HeliaAotOptions, compile_model

compiled = compile_model(model, example_inputs, HeliaAotOptions(output_dir="build"), name="kws")
# build/kws/            heliaAOT NSX module: kws_model_init(&ctx) once, then fill ctx.inputs, kws_model_run(&ctx)
# build/kws/kws_io.json int8 boundary: per input/output scale, zero point, logical shape, dim order
compiled.inputs[0].quantize(x)   # float, logical shape -> int8 bytes in the module's layout
```

- **Same code as the delegate.** It runs the same lowering, frontend and heliaAOT rules, so it is bit-exact
  with the non-delegated ExecuTorch program. `aot/tests/test_helia_aot.py` checks all 26 cases on the host
  through `tests/helia_aot_host/standalone_main.c`, which calls the module as an application does.
- **Input.** `model` can be an eager module or float program (quantized with the ns quantizer), or an
  already-quantized `ExportedProgram`, for example `torch.export.load()` of a saved `quantize()` result.
  `program_to_air()` stops before code generation and returns the AIR plus the boundary.
- **Fully covered models only.** If any operator would stay in ExecuTorch, `compile_model()` raises
  `NotFullyDelegableError` naming each node and why; use `export(..., helia_aot=...)` for those models.
- **The boundary is the application's.** The module takes and returns int8 tensors in their physical (dim
  order) layout, NHWC for channels_last; `IoTensor.quantize()` / `dequantize()` convert.
- **What it saves.** On an Apollo510 EVB the KWS DS-CNN as a standalone module took 1,622,688 cycles and
  152,788 bytes of flash, against 1,625,907 cycles and 195,776 bytes through the ExecuTorch runtime and the
  delegate (int8 I/O, no `cortex_m` operators registered). The difference is the runtime (about 26 KB), the PTE,
  the delegate's dispatch and contract checks, and its input/output copies.
- **What it gives up.** The model is firmware code rather than a loadable PTE, and there is no ExecuTorch
  method API or ExecuTorch tooling.

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
