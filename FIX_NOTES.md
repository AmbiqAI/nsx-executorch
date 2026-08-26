# Fix notes: nsx-executorch#11

`dim_order_ops::_clone_dim_order` — the memory-format copy the dim-order
pipeline inserts at method/op boundaries — had no Cortex-M lowering and
always stayed on the portable kernel: the only unaccelerated op across the
MLPerf Tiny hardware-validation fixtures (KWS DS-CNN carries 2 invocations
per inference on its int8 input path). Follow-up to #10.

## Fix

Adds `NsCloneDimOrderRewritePass` (`aot/nsx_cortex_m/passes_ns.py`), wired
into `NS_PASS_LIST` only (`aot/nsx_cortex_m/pass_manager_ns.py`). It
recognizes an int8 `_clone_dim_order` whose input/output dim_order already
match (a no-op relayout — the case a real KWS-shaped export actually
produces) and rewrites it to `cortex_m::transpose` with the identity
permutation.

## Why scoped to the identity-dim_order case

`_clone_dim_order` is shape-preserving — only the tensor's *physical*
dim_order changes, never its logical shape. `cortex_m::transpose` is a
real permute: its runtime kernel (`op_transpose.cpp`'s `transpose_out`)
reads `out.size(i)` directly to build the CMSIS-NN dims struct, so a
non-identity permutation genuinely reshapes the tensor. Consumers
downstream of a channels_last activation — e.g.
`cortex_m::quantized_depthwise_conv2d`, whose kernel reads `input.size(1)`
as the channel count regardless of physical layout (see
`quantized_depthwise_conv2d_meta`'s `memory_format=torch.channels_last`
output construction) — require the original logical shape to survive
unchanged. A literal op-swap for a genuinely differing dim_order
(contiguous -> channels_last) would silently corrupt that convolution.

Only the identity-permutation case is provably safe with the *existing*
`cortex_m::transpose` op, and it's the case a real KWS DS-CNN export
actually produces. A genuine relayout would need a dedicated kernel and is
left as a documented follow-up rather than shipped as an unverified "fix"
(see the pass's docstring for the full reasoning).

## Testing

- New `aot/tests/test_clone_dim_order.py`: exports a KWS DS-CNN-shaped
  model (mirrors `executorch/examples/models/mlperf_tiny/ds_cnn.py`) with
  `int8_io=True` and asserts the 2 `_clone_dim_order` nodes become 2
  `cortex_m::transpose` nodes with zero portable fallback ops, and that
  the `arm` kernel provider is unaffected (clones stay portable, no
  `cortex_m_ns`/transpose ops).
- Full suite: `python -m pytest aot/tests -q` — 46 passed (44 existing +
  2 new), verified inside the `.devcontainer/` added alongside this fix
  (mirrors the `aot-tests` CI job: Python 3.12, torch 2.12, executorch
  1.3.1, no ARM toolchain needed for this Python-side lowering pass).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
