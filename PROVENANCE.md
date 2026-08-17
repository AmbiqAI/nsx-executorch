# Source provenance

This repository is an NSX adapter, not an ExecuTorch fork.

| Dependency | Repository | Revision | Role |
| --- | --- | --- | --- |
| ExecuTorch | `pytorch/executorch` | `3a97429b0ce0c192861fc3e3729fb81432fd22cf` (`v1.3.0`) | Unmodified runtime and Cortex-M backend |
| torchgen | `pytorch/pytorch` | `70d99e998b4955e0049d13a98d77ae1b14db1f45` (`v2.11.0`) | Offline host-side selective-kernel generation |

`ExecuTorch` is a pinned Git submodule. `torchgen` is vendored under
`tools/python/` because ExecuTorch's CMake code generation imports it at
configure and build time, while ordinary NSX installations do not install the
full PyTorch host package.

## CMSIS-NN providers

Neither Arm CMSIS-NN nor CMSIS 6 is vendored in this repository. Both build
providers are resolved as NeuralSPOT-X (NSX) module dependencies at configure
time, selected by `NSX_EXECUTORCH_CMSIS_NN_PROVIDER`:

| Provider | NSX module | Repository | Exported target | Role |
| --- | --- | --- | --- | --- |
| `arm` (default) | `arm-cmsis-nn` | `AmbiqAI/arm-cmsis-nn` | `nsx::arm_cmsis_nn` | Thin wrapper that itself vendors upstream `ARM-software/CMSIS-NN` and `ARM-software/CMSIS_6` under its own `external/`; see that repository's own PROVENANCE.md for exact upstream revisions |
| `ns` | `nsx-cmsis-nn` | `AmbiqAI/ns-cmsis-nn` (project `ns-cmsis-nn`) | `nsx::cmsis_nn` (separate `nsx/` entry point; not used here) | Ambiq-optimized, source-compatible drop-in replacement for CMSIS-NN |

For both providers, `nsx-executorch` sets stock ExecuTorch's own
`CMSIS_NN_LOCAL_PATH` (and, for `arm`, `CMSIS_PATH`) to point at the resolved
module's source, so ExecuTorch's unmodified `backends/cortex_m/CMakeLists.txt`
remains the single place that creates the `cmsis-nn` CMake target — no
duplicate target, no ExecuTorch source patch, and no network `FetchContent`
fallback.

- For `arm`, `CMSIS_NN_LOCAL_PATH`/`CMSIS_PATH` point directly at
  `arm-cmsis-nn`'s vendored `external/CMSIS-NN` and `external/CMSIS_6`. After
  ExecuTorch creates `cmsis-nn` from that source, `nsx-executorch` also adds
  `arm-cmsis-nn`'s own `CMakeLists.txt`, which detects the existing
  `cmsis-nn` target (it is written as `if(NOT TARGET cmsis-nn) ... endif()`),
  skips re-creating it, and just attaches NSX board flags plus the
  `nsx::arm_cmsis_nn` alias.
- For `ns`, `CMSIS_NN_LOCAL_PATH` points at the resolved `ns-cmsis-nn`
  package's **root** `CMakeLists.txt`, which is a deliberate
  upstream-compatible drop-in that creates the `cmsis-nn` target directly
  from Ambiq's optimized sources. `ns-cmsis-nn` also ships a separate
  `nsx/CMakeLists.txt` entry point (exporting `nsx::cmsis_nn`, package
  `nsx_cmsis_nn`) for other NSX consumers; that entry point compiles an
  unrelated target and is intentionally not used by `nsx-executorch`, since
  doing so would double-compile the same kernels for no benefit. The `ns`
  provider is required to implement the same headers, function signatures,
  behavior, and CMake contract as upstream CMSIS-NN. No ExecuTorch source
  patch accommodates provider-specific APIs.

Module resolution never guesses sibling directories. It uses an explicit
cache-variable override (`NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT` /
`NSX_EXECUTORCH_NS_CMSIS_NN_ROOT`) when set, or the same
`NSX_ROOT`/`NSX_APP_MODULE_DIR_<module>` contract the NSX tooling itself uses
to vendor module checkouts when consumed inside a bootstrapped NSX app.
