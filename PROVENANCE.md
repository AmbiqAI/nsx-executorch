# Source provenance

This repository is an NSX adapter, not an ExecuTorch fork.

| Dependency | Repository | Revision | Role |
| --- | --- | --- | --- |
| ExecuTorch | `pytorch/executorch` | `3a97429b0ce0c192861fc3e3729fb81432fd22cf` (`v1.3.0`) | Unmodified runtime and Cortex-M backend |
| Arm CMSIS-NN | `ARM-software/CMSIS-NN` | `d933672e7ca97eec70ef43230baee7b20c2a28ae` | Stock Cortex-M kernel provider |
| CMSIS 6 | `ARM-software/CMSIS_6` | `7f62ddc8ab8e9af22039912b8f9f46a9290f49ba` | Cortex-M core headers |
| torchgen | `pytorch/pytorch` | `70d99e998b4955e0049d13a98d77ae1b14db1f45` (`v2.11.0`) | Offline host-side selective-kernel generation |

The three source dependencies are pinned Git submodules. `torchgen` is
vendored under `tools/python/` because ExecuTorch's CMake code generation
imports it at configure and build time, while ordinary NSX installations do
not install the full PyTorch host package.

The `ns` provider is intentionally not vendored here. It resolves an app-local
`ns-cmsis-nn` module and requires that module to implement the same headers,
function signatures, behavior, and CMake contract as upstream CMSIS-NN. No
ExecuTorch source patch accommodates provider-specific APIs.
