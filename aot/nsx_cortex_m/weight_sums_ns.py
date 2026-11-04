# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# ns-cmsis-nn per-output-channel weight sums, computed at export time.
#
# ns-cmsis-nn's MVE convolution kernels take a caller-supplied buffer of
#
#     weight_sum[j] = input_offset * sum(weights of output channel j) + bias[j]
#
# (int32, wrapping), the layout arm_convolve_weight_sum() and
# arm_depthwise_convolve_weight_sum() fill. The kernels only read it and allow
# reusing it while filter, bias and input_offset are unchanged, so for the
# constant weights of an exported model it can be computed once here and
# stored in the PTE instead of being recomputed on every inference.
#
# WEIGHT_SUM_FORMAT versions that layout. The runtime kernels
# (ops-ns/op_quantized_conv2d_ws.cpp, kWeightSumFormat) reject any other
# value, so a provider that changes the layout fails loudly instead of
# computing wrong outputs. Bump both together.

from typing import Optional

import torch

WEIGHT_SUM_FORMAT = 1

_U32 = 1 << 32
_I32_MIN = -(1 << 31)


def _wrap_int32(values: torch.Tensor) -> torch.Tensor:
    """Two's-complement wrap of int64 values to int32, as the C kernels do.

    Their arithmetic is adds and multiplies modulo 2^32, so wrapping the
    exact int64 result once gives the same bits.
    """
    return ((values.to(torch.int64) - _I32_MIN) % _U32 + _I32_MIN).to(torch.int32)


def _finish(
    channel_sums: torch.Tensor, bias: Optional[torch.Tensor], input_offset: int
) -> torch.Tensor:
    sums = channel_sums.to(torch.int64) * int(input_offset)
    if bias is not None:
        if bias.dtype != torch.int32 or bias.dim() != 1 or bias.numel() != sums.numel():
            raise ValueError(
                f"bias must be int32 [{sums.numel()}], got {bias.dtype} {list(bias.shape)}"
            )
        sums = sums + bias.to(torch.int64)
    return _wrap_int32(sums)


def conv_weight_sums(
    weight_ohwi: torch.Tensor, bias: Optional[torch.Tensor], input_offset: int
) -> torch.Tensor:
    """Weight sums for a cortex_m conv2d weight in OHWI ([C_OUT, H, W, C_IN]).

    Matches arm_convolve_weight_sum(): one row of H*W*C_IN weights per
    output channel.
    """
    if weight_ohwi.dtype != torch.int8 or weight_ohwi.dim() != 4:
        raise ValueError(f"expected an int8 OHWI weight, got {weight_ohwi.dtype} {list(weight_ohwi.shape)}")
    channel_sums = weight_ohwi.to(torch.int64).reshape(weight_ohwi.shape[0], -1).sum(dim=1)
    return _finish(channel_sums, bias, input_offset)


def depthwise_weight_sums(
    weight_ihwo: torch.Tensor, bias: Optional[torch.Tensor], input_offset: int
) -> torch.Tensor:
    """Weight sums for a cortex_m depthwise weight in IHWO ([1, H, W, C_OUT]).

    Matches arm_depthwise_convolve_weight_sum(): per output channel, the sum
    over the H*W filter taps (any depth multiplier is already folded into
    C_OUT).
    """
    if weight_ihwo.dtype != torch.int8 or weight_ihwo.dim() != 4 or weight_ihwo.shape[0] != 1:
        raise ValueError(f"expected an int8 IHWO [1, H, W, C_OUT] weight, got {weight_ihwo.dtype} {list(weight_ihwo.shape)}")
    channel_sums = weight_ihwo.to(torch.int64).reshape(-1, weight_ihwo.shape[3]).sum(dim=0)
    return _finish(channel_sums, bias, input_offset)
