# CMSIS-NN provider fixtures

These directories are small, NSX-authored stand-ins for the two real CMSIS-NN
NSX modules that `nsx-executorch` can consume. They are **not** vendored
copies of any upstream or Ambiq source; they exist only so both provider
code paths in the root `CMakeLists.txt` can be configured (and, where
feasible, built) in CI without network access to the real module
repositories.

- `arm-cmsis-nn/` mirrors the CMake contract of `AmbiqAI/arm-cmsis-nn`: a
  root `CMakeLists.txt` that requires `NSX_BOARD_FLAGS_TARGET`, idempotently
  creates (or reuses) a `cmsis-nn` target from a vendored
  `external/CMSIS-NN` stub, and exports `nsx::arm_cmsis_nn`.
- `ns-cmsis-nn/` mirrors the CMake contract of `AmbiqAI/ns-cmsis-nn`'s
  **root** package (not its separate `nsx/` entry point): a
  drop-in that creates the `cmsis-nn` target directly.

Point `NSX_EXECUTORCH_ARM_CMSIS_NN_ROOT` / `NSX_EXECUTORCH_NS_CMSIS_NN_ROOT`
at these directories to exercise each provider without a real module
checkout; see `tests/smoke` and `tests/smoke-ns`.
