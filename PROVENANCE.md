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
| `arm` (default) | `arm-cmsis-nn` | `AmbiqAI/arm-cmsis-nn` | `nsx::arm_cmsis_nn` | `v0.1.0`, commit `6d21a6f821fb72541173a6c4d05d83329fa74f7c`; thin wrapper around its pinned upstream CMSIS-NN and CMSIS 6 submodules |
| `ns` | `nsx-cmsis-nn` | `AmbiqAI/ns-cmsis-nn` (project `ns-cmsis-nn`) | `nsx::cmsis_nn` | `v7.29.2`, commit `631726420b04860a5c4236956a3741ff5a96bd7f`; Ambiq-optimized kernels with the NSX module entry point under `nsx/` |

`nsx-executorch` sets stock ExecuTorch's `CMSIS_NN_LOCAL_PATH` to a local
provider adapter. The adapter reuses an existing provider target when NSX
already configured it, or configures the selected source for standalone use.
This prevents ExecuTorch's non-idempotent Cortex-M CMake from adding the same
provider twice. `FETCHCONTENT_FULLY_DISCONNECTED` is forced on, so CMake
cannot use ExecuTorch's network fallback. Stock ExecuTorch install/export
rules are suppressed because this repository publishes a build-tree NSX
target rather than a standalone installed ExecuTorch SDK.

- For `arm`, the adapter reuses the real `cmsis-nn` target exported through
  `nsx::arm_cmsis_nn`. In standalone mode it adds the provider's pinned
  `external/CMSIS-NN` source and then adds the wrapper to publish the NSX
  alias.
- For `ns`, NSX resolves the module directory to
  `modules/ns-cmsis-nn/nsx`, while the upstream-compatible source root is its
  parent. `nsx-executorch` normalizes either form. When `nsx::cmsis_nn`
  already exists, the adapter creates a single interface `cmsis-nn` bridge;
  standalone mode adds the repository-root package directly.

NS-CMSIS-NN's stock wrapper signatures diverged at commit
`cd5f6acfadc29c53145cd2154be3e6f9b58d6631` (first released in v7.2.0):
`arm_convolve_wrapper_s8`, `arm_depthwise_conv_wrapper_s8`, and
`arm_transpose_conv_wrapper_s8` require an additional weight-sum context.
The last stock-signature revision, v7.1.0
(`d9a614f666fa595bef58778f1a547e4610310e59`), predates NSX module support, so
it is not a viable provider pin. The private compatibility header in this
repository extends ExecuTorch's existing temporary buffers, precomputes the
required sums, and calls the v7.29.2 APIs. The ExecuTorch submodule remains
unmodified.

Module resolution never guesses sibling projects. It uses an explicit
cache-variable override (`NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT` /
`NSX_EXECUTORCH_NS_CMSIS_NN_ROOT`) when set, or the same
`NSX_ROOT`/`NSX_APP_MODULE_DIR_<module>` contract the NSX tooling itself uses
to vendor module checkouts. The selected optional provider must be a direct
app dependency listed before `nsx-executorch`.
