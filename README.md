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

- `arm` uses the pinned, stock CMSIS-NN submodule.
- `ns` uses an app-local `ns-cmsis-nn` checkout. That checkout must be a true
  source-compatible drop-in for upstream CMSIS-NN, including its CMake target.

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

Clone with the pinned source dependencies:

```sh
git clone --recurse-submodules https://github.com/AmbiqAI/nsx-executorch.git
```

For a minimal Cortex-M checkout, initialize the three top-level dependencies
and the ExecuTorch submodules used by its stock CMake configuration:

```sh
git submodule update --init external/executorch external/CMSIS-NN external/CMSIS_6
git -C external/executorch submodule update --init \
  backends/xnnpack/third-party/FXdiv \
  third-party/flatbuffers third-party/flatcc third-party/gflags third-party/json
```

Then verify the exact revisions and configure the profiling-enabled smoke
project:

```sh
python3 tests/verify_source_pins.py
cmake -S tests/smoke -B build/configure-smoke -G Ninja
```

The `ns` provider intentionally fails unless `ns-cmsis-nn` satisfies the same
source and CMake contract as upstream CMSIS-NN. Provider-specific ExecuTorch
patches belong neither here nor in the model export flow.
