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
- `ns` resolves the `nsx-cmsis-nn` NSX module (project repo
  `AmbiqAI/ns-cmsis-nn`). That checkout's root must be a true
  source-compatible drop-in for upstream CMSIS-NN, including its CMake
  target — this is the same contract as before, just resolved as a real NSX
  module dependency instead of an app-local checkout.

Neither CMSIS-NN nor CMSIS 6 is vendored directly in this repository anymore;
both are consumed as NSX module dependencies. When this package is added from
inside a bootstrapped NSX app, module resolution happens automatically. For
standalone use (development, CI, or direct `add_subdirectory()`), point the
provider's module root explicitly:

```cmake
# provider=arm
set(NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT "/path/to/arm-cmsis-nn" CACHE PATH "" FORCE)
# provider=ns
set(NSX_EXECUTORCH_NS_CMSIS_NN_ROOT "/path/to/ns-cmsis-nn" CACHE PATH "" FORCE)
```

Both providers consume the same stock Cortex-M operators and the same PTE.
Provider selection never changes operator schemas or ExecuTorch lowering.

Enable layer callbacks with:

```cmake
set(NSX_EXECUTORCH_ENABLE_PROFILING ON CACHE BOOL "" FORCE)
```

The callbacks report ExecuTorch chain and instruction identifiers. A caller
such as Helia Profiler can map those identifiers to exported model metadata
and sample PMU counters around every operator.

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
```

The `ns` provider intentionally fails unless `ns-cmsis-nn` satisfies the same
source and CMake contract as upstream CMSIS-NN. Provider-specific ExecuTorch
patches belong neither here nor in the model export flow.
