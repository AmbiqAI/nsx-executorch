# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# `cortex_m_ns` operator library: torch.library definitions, fake metas and
# composite reference implementations for the Ambiq ns-cmsis-nn additional
# kernels. The reference implementations mirror the pinned kernels' integer
# math bit-exactly (requantize, offsets, clamping) so AOT numerics tests are
# meaningful.
#
# Runtime schemas live in ops-ns/operators_ns.yaml and must stay in sync.

import torch
from executorch.backends.cortex_m.passes.passes_utils import (
    is_channel_broadcast,
    requantize_cmsis,
    SHIFT_INT8,
)
from torch.library import impl, Library, register_fake

# Sentinel emitted by the AOT lowering when the requantization is an exact
# identity: quantize_multiplier_aot(1.0). The runtime uses it to select the
# arm_clamp_s8 fast path.
IDENTITY_REQUANT_MULTIPLIER = 1 << 30
IDENTITY_REQUANT_SHIFT = 1

ns_lib = Library("cortex_m_ns", "DEF")

# ===================================================================
# quantized_sub (arm_elementwise_sub_s8)
# ===================================================================

ns_lib.define(
    "quantized_sub("
    "Tensor self, int self_zero_point, int self_multiplier, int self_shift, "
    "Tensor other, int other_zero_point, int other_multiplier, int other_shift, "
    "int output_zero_point, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max) -> Tensor"
)
ns_lib.define(
    "quantized_sub.out("
    "Tensor self, int self_zero_point, int self_multiplier, int self_shift, "
    "Tensor other, int other_zero_point, int other_multiplier, int other_shift, "
    "int output_zero_point, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m_ns::quantized_sub")
def quantized_sub_meta(
    self: torch.Tensor,
    self_zero_point: int,
    self_multiplier: int,
    self_shift: int,
    other: torch.Tensor,
    other_zero_point: int,
    other_multiplier: int,
    other_shift: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "cortex_m_ns quantized_sub: broadcasting is not supported except for "
        f"the channel dim — got self.shape={self.shape}, other.shape={other.shape}"
    )
    output_tensor = self if self.numel() > other.numel() else other
    return torch.empty_like(output_tensor)


@impl(ns_lib, "quantized_sub", "CompositeExplicitAutograd")
def quantized_sub_impl(
    self: torch.Tensor,
    self_zero_point: int,
    self_multiplier: int,
    self_shift: int,
    other: torch.Tensor,
    other_zero_point: int,
    other_multiplier: int,
    other_shift: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "cortex_m_ns quantized_sub: broadcasting is not supported except for "
        f"the channel dim — got self.shape={self.shape}, other.shape={other.shape}"
    )
    self_shifted = (self.to(torch.int32) - self_zero_point) << SHIFT_INT8
    self_fp = requantize_cmsis(self_shifted, self_multiplier, self_shift)

    other_shifted = (other.to(torch.int32) - other_zero_point) << SHIFT_INT8
    other_fp = requantize_cmsis(other_shifted, other_multiplier, other_shift)

    result_fp = self_fp - other_fp
    result_quantized = requantize_cmsis(result_fp, output_multiplier, output_shift)
    return torch.clamp(
        result_quantized + output_zero_point, activation_min, activation_max
    ).to(torch.int8)


# ===================================================================
# quantized_hardswish (arm_hard_swish_precise_s8)
# ===================================================================

ns_lib.define(
    "quantized_hardswish("
    "Tensor self, int input_zero_point, int output_zero_point, "
    "int output_multiplier, int output_shift, "
    "int relu_q3, int relu_q6, int prescale) -> Tensor"
)
ns_lib.define(
    "quantized_hardswish.out("
    "Tensor self, int input_zero_point, int output_zero_point, "
    "int output_multiplier, int output_shift, "
    "int relu_q3, int relu_q6, int prescale, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m_ns::quantized_hardswish")
def quantized_hardswish_meta(
    self: torch.Tensor,
    input_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    relu_q3: int,
    relu_q6: int,
    prescale: int,
) -> torch.Tensor:
    return torch.empty_like(self)


@impl(ns_lib, "quantized_hardswish", "CompositeExplicitAutograd")
def quantized_hardswish_impl(
    self: torch.Tensor,
    input_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    relu_q3: int,
    relu_q6: int,
    prescale: int,
) -> torch.Tensor:
    x = self.to(torch.int32) - input_zero_point
    xr = torch.clamp(x + relu_q3, 0, relu_q6)
    if prescale > 0:
        # arm_nn_nonneg_divide_by_pot_s32: rounding right shift.
        xr = (xr + (1 << (prescale - 1))) >> prescale
    y = requantize_cmsis(x * xr, output_multiplier, output_shift)
    return torch.clamp(y + output_zero_point, -128, 127).to(torch.int8)


# ===================================================================
# quantized_mean (arm_mean_s8)
# ===================================================================

ns_lib.define(
    "quantized_mean("
    "Tensor self, int input_zero_point, int[] axis_mask, bool keepdim, "
    "int output_zero_point, int output_multiplier, int output_shift) -> Tensor"
)
ns_lib.define(
    "quantized_mean.out("
    "Tensor self, int input_zero_point, int[] axis_mask, bool keepdim, "
    "int output_zero_point, int output_multiplier, int output_shift, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


def _mean_reduce_dims(rank: int, axis_mask) -> list[int]:
    # axis_mask is a 4-long binary mask over the right-aligned (n, h, w, c)
    # padding of the logical shape.
    offset = 4 - rank
    return [i - offset for i in range(4) if axis_mask[i] and i >= offset]


@register_fake("cortex_m_ns::quantized_mean")
def quantized_mean_meta(
    self: torch.Tensor,
    input_zero_point: int,
    axis_mask,
    keepdim: bool,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
) -> torch.Tensor:
    assert self.dim() <= 4, "cortex_m_ns quantized_mean: rank must be <= 4"
    assert len(axis_mask) == 4, "cortex_m_ns quantized_mean: axis_mask must be 4-long"
    dims = _mean_reduce_dims(self.dim(), axis_mask)
    shape = list(self.shape)
    if keepdim:
        for d in dims:
            shape[d] = 1
    else:
        shape = [s for i, s in enumerate(shape) if i not in dims]
    return torch.empty(shape, dtype=self.dtype, device=self.device)


@impl(ns_lib, "quantized_mean", "CompositeExplicitAutograd")
def quantized_mean_impl(
    self: torch.Tensor,
    input_zero_point: int,
    axis_mask,
    keepdim: bool,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
) -> torch.Tensor:
    assert self.dim() <= 4, "cortex_m_ns quantized_mean: rank must be <= 4"
    dims = _mean_reduce_dims(self.dim(), axis_mask)
    # The kernel accumulates raw values plus the (negated) zero point, then
    # requantizes with the reduce count folded into multiplier/shift AOT.
    acc = (self.to(torch.int64) - input_zero_point).sum(dim=dims, keepdim=keepdim)
    y = requantize_cmsis(acc, output_multiplier, output_shift)
    return torch.clamp(y + output_zero_point, -128, 127).to(torch.int8)


# ===================================================================
# quantized_relu (arm_relu_generic_s8 / arm_clamp_s8)
# ===================================================================

ns_lib.define(
    "quantized_relu("
    "Tensor self, int input_zero_point, int output_zero_point, "
    "int output_multiplier, int output_shift, "
    "int activation_min, int activation_max) -> Tensor"
)
ns_lib.define(
    "quantized_relu.out("
    "Tensor self, int input_zero_point, int output_zero_point, "
    "int output_multiplier, int output_shift, "
    "int activation_min, int activation_max, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m_ns::quantized_relu")
def quantized_relu_meta(
    self: torch.Tensor,
    input_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    return torch.empty_like(self)


@impl(ns_lib, "quantized_relu", "CompositeExplicitAutograd")
def quantized_relu_impl(
    self: torch.Tensor,
    input_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    # arm_relu_generic_s8: subtract input offset, requantize, add output
    # offset, clamp to the activation bounds (already in the output quantized
    # domain). The identity-sentinel arm_clamp_s8 fast path is numerically
    # identical because requantize with (1<<30, 1) is exact.
    x = self.to(torch.int32) - input_zero_point
    y = requantize_cmsis(x, output_multiplier, output_shift) + output_zero_point
    act_min = max(-128, activation_min)
    act_max = min(127, activation_max)
    return torch.clamp(y, act_min, act_max).to(torch.int8)


# ===================================================================
# quantized_leaky_relu (arm_leaky_relu_s8)
# ===================================================================

ns_lib.define(
    "quantized_leaky_relu("
    "Tensor self, int input_zero_point, int output_zero_point, "
    "int alpha_multiplier, int alpha_shift, "
    "int identity_multiplier, int identity_shift) -> Tensor"
)
ns_lib.define(
    "quantized_leaky_relu.out("
    "Tensor self, int input_zero_point, int output_zero_point, "
    "int alpha_multiplier, int alpha_shift, "
    "int identity_multiplier, int identity_shift, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m_ns::quantized_leaky_relu")
def quantized_leaky_relu_meta(
    self: torch.Tensor,
    input_zero_point: int,
    output_zero_point: int,
    alpha_multiplier: int,
    alpha_shift: int,
    identity_multiplier: int,
    identity_shift: int,
) -> torch.Tensor:
    return torch.empty_like(self)


@impl(ns_lib, "quantized_leaky_relu", "CompositeExplicitAutograd")
def quantized_leaky_relu_impl(
    self: torch.Tensor,
    input_zero_point: int,
    output_zero_point: int,
    alpha_multiplier: int,
    alpha_shift: int,
    identity_multiplier: int,
    identity_shift: int,
) -> torch.Tensor:
    x = self.to(torch.int32) - input_zero_point
    identity = requantize_cmsis(x, identity_multiplier, identity_shift)
    alpha = requantize_cmsis(x, alpha_multiplier, alpha_shift)
    y = torch.where(x >= 0, identity, alpha) + output_zero_point
    return torch.clamp(y, -128, 127).to(torch.int8)


# ===================================================================
# leaky_relu_stub (AOT-only placeholder, never reaches the runtime)
#
# aten.leaky_relu cannot be re-traced with an int8 fake input and a
# float negative_slope (torch refuses the scalar cast), which breaks
# FoldAndAnnotateQParamsPass. NsLeakyReluCapturePass swaps the aten op
# for this dtype-agnostic stub before folding; NsQuantizedOpFusionPass
# then either fuses it into quantized_leaky_relu or reverts it to
# aten.leaky_relu. No .out variant on purpose: the stub must never
# survive to_executorch.
# ===================================================================

ns_lib.define("leaky_relu_stub(Tensor self, float negative_slope) -> Tensor")


@register_fake("cortex_m_ns::leaky_relu_stub")
def leaky_relu_stub_meta(self: torch.Tensor, negative_slope: float) -> torch.Tensor:
    return torch.empty_like(self)


@impl(ns_lib, "leaky_relu_stub", "CompositeExplicitAutograd")
def leaky_relu_stub_impl(self: torch.Tensor, negative_slope: float) -> torch.Tensor:
    if self.dtype.is_floating_point:
        return torch.nn.functional.leaky_relu(self, negative_slope)
    return torch.nn.functional.leaky_relu(self.to(torch.float32), negative_slope).to(
        self.dtype
    )


# ===================================================================
# mean_stub (AOT-only placeholder, never reaches the runtime)
#
# Like leaky_relu_stub: aten.mean.dim cannot be re-traced with an int8
# fake input ("could not infer output dtype"), which breaks
# FoldAndAnnotateQParamsPass once q/dq are folded. The stub's fake meta
# preserves the input dtype.
# ===================================================================

ns_lib.define("mean_stub(Tensor self, int[] dims, bool keepdim) -> Tensor")


def _mean_stub_output_shape(self: torch.Tensor, dims, keepdim: bool) -> list[int]:
    rank = self.dim()
    reduce = {d % rank for d in dims}
    shape = []
    for i, s in enumerate(self.shape):
        if i in reduce:
            if keepdim:
                shape.append(1)
        else:
            shape.append(int(s))
    return shape


@register_fake("cortex_m_ns::mean_stub")
def mean_stub_meta(self: torch.Tensor, dims, keepdim: bool) -> torch.Tensor:
    return torch.empty(
        _mean_stub_output_shape(self, dims, keepdim), dtype=self.dtype
    )


@impl(ns_lib, "mean_stub", "CompositeExplicitAutograd")
def mean_stub_impl(self: torch.Tensor, dims, keepdim: bool) -> torch.Tensor:
    if self.dtype.is_floating_point:
        return torch.mean(self, dim=list(dims), keepdim=keepdim)
    return (
        torch.mean(self.to(torch.float32), dim=list(dims), keepdim=keepdim)
        .round()
        .to(self.dtype)
    )
