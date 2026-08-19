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

The callbacks report ExecuTorch chain and instruction identifiers. A caller
such as Helia Profiler can map those identifiers to exported model metadata
and sample PMU counters around every operator.

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
result.write_pte("model.pte")
print(result.edge_ops)                  # lowered operator histogram
print(result.portable_select_ops_list)  # NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST value
```

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
