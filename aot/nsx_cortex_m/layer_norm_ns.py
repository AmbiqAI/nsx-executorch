# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Layer normalization for ns-cmsis-nn's arm_layer_norm_s8 / arm_layer_norm_s16.
#
# export_params() folds a float layer norm's affine parameters and its input/output quantization into the kernel's
# integer parameters; layer_norm_int() is the kernel's exact integer algorithm, vectorized in torch, which the
# cortex_m_ns::quantized_layer_norm reference implementation runs on the host.
#
# Both mirror ns-cmsis-nn's Tests/UnitTest/layer_norm_reference.py, which defines the algorithm (the C kernels
# match it bit for bit); aot/tests/test_layer_norm.py checks this file against that reference and against the C
# kernel itself. In short, per row of N values x:
#
#   D = N * sum(x^2) - sum(x)^2 + epsilon                     (exact; N^2 * (variance + eps))
#   (m, L) = row_multiplier(D)                                 (2^20 / sqrt(D) ~= m * 2^L / 2^31)
#   z = qrdmulh((N << L) * x - (S << L), m)                    (normalized, 20 fraction bits)
#   y = clamp(rshr(qrdmulh(z, gamma[c]) + bias[c], output_shift))
#
# qrdmulh(a, b) = (a * b + 2^30) >> 31 and rshr(v, T) = (v + 2^(T - 1)) >> T.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

FRAC_BITS = 20
NEWTON_STEPS = 3
MAX_ROW_SIZE = {torch.int8: 65536, torch.int16: 32767}
EPSILON_MAX = (1 << 62) - 1 - (1 << 60)
SEED_TABLE = [int(round(2**31 / math.sqrt((i + 0.5) / 8))) for i in range(8, 32)]


class LayerNormNotSupported(ValueError):
    """The layer norm's parameters fall outside what the kernel represents (the export keeps it in float)."""


@dataclass(frozen=True)
class LayerNormParams:
    epsilon: int
    output_shift: int
    gamma: torch.Tensor  # int32 [N]
    bias: torch.Tensor  # int32 [N]
    activation_min: int
    activation_max: int


def export_params(
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    eps: float,
    input_scale: float,
    output_scale: float,
    output_zero_point: int,
    row_size: int,
    dtype: torch.dtype,
) -> LayerNormParams:
    """Kernel parameters for a layer norm over the last `row_size` values (gamma/beta None = identity)."""
    if dtype not in MAX_ROW_SIZE:
        raise LayerNormNotSupported(f"{dtype} activations")
    if not 1 <= row_size <= MAX_ROW_SIZE[dtype]:
        raise LayerNormNotSupported(f"row size {row_size} outside [1, {MAX_ROW_SIZE[dtype]}] for {dtype}")
    g = torch.ones(row_size, dtype=torch.float64) if gamma is None else gamma.detach().to(torch.float64).reshape(-1)
    b = torch.zeros(row_size, dtype=torch.float64) if beta is None else beta.detach().to(torch.float64).reshape(-1)
    if g.numel() != row_size or b.numel() != row_size:
        raise LayerNormNotSupported("gamma and beta need one value per element of the row")
    epsilon = min(int(round(row_size * row_size * eps / (input_scale * input_scale))), EPSILON_MAX)
    ratio = g / output_scale
    amax = float(ratio.abs().max())
    exponent = max(math.frexp(amax)[1] if amax > 0 else FRAC_BITS - 31, FRAC_BITS - 31)
    bmax = float((b / output_scale).abs().max()) + abs(int(output_zero_point))
    if bmax > 0:
        exponent = max(exponent, FRAC_BITS - (29 - math.frexp(bmax)[1]))
    output_shift = FRAC_BITS - exponent
    if output_shift < 1:
        raise LayerNormNotSupported(f"|gamma| / output_scale up to {amax:.4g} needs < 2^{FRAC_BITS - 1}")
    gq = torch.clamp(torch.round(ratio * 2.0 ** (31 - exponent)), -(2**31 - 1), 2**31 - 1).to(torch.int64)
    bq = torch.round(b / output_scale * 2.0**output_shift).to(torch.int64) + (int(output_zero_point) << output_shift)
    if bool((bq.abs() >= 2**30).any()):
        raise LayerNormNotSupported("beta / output_scale too large for the output shift")
    info = torch.iinfo(dtype)
    return LayerNormParams(epsilon=epsilon, output_shift=output_shift, gamma=gq.to(torch.int32),
                           bias=bq.to(torch.int32), activation_min=info.min, activation_max=info.max)


def _row_multiplier(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(m, L) per row for d (int64, [rows]) in [0, 2^62): see ns-cmsis-nn arm_nn_layer_norm_row_multiplier()."""
    zero = d == 0
    dd = torch.where(zero, torch.ones_like(d), d)
    top_bit = torch.floor(torch.log2(dd.to(torch.float64))).to(torch.int64)
    # log2 in float64 can be off by one next to a power of two; fix it exactly.
    top_bit = torch.where(torch.bitwise_right_shift(dd, top_bit) == 0, top_bit - 1, top_bit)
    top_bit = torch.where(torch.bitwise_right_shift(dd, top_bit + 1) != 0, top_bit + 1, top_bit)
    a = torch.bitwise_right_shift(61 - top_bit, 1)
    # dd << 2a stays below 2^62 (and inside int64); then x = (dd << 2a) >> 30 is in [2^30, 2^32).
    x = torch.bitwise_right_shift(torch.bitwise_left_shift(dd, 2 * a), 30)
    seed = torch.tensor(SEED_TABLE, dtype=torch.int64)
    y = seed[torch.bitwise_right_shift(x, 27) - 8]
    for _ in range(NEWTON_STEPS):
        y2 = torch.bitwise_right_shift(y * y, 31)
        xy2 = torch.bitwise_right_shift(x * y2, 30)
        y = y + torch.bitwise_right_shift(y * ((1 << 31) - xy2), 32)
    y = torch.clamp(y, max=2**31 - 1)
    shift = FRAC_BITS + a - 30
    neg = shift < 0
    rs = torch.where(neg, -shift, torch.zeros_like(shift))
    y = torch.where(neg, torch.bitwise_right_shift(y + torch.bitwise_left_shift(torch.ones_like(rs), rs - 1), rs), y)
    shift = torch.where(neg, torch.zeros_like(shift), shift)
    return torch.where(zero, torch.zeros_like(y), y), torch.where(zero, torch.zeros_like(shift), shift)


def _qrdmulh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_right_shift(a * b + (1 << 30), 31)


def layer_norm_int(x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor, epsilon: int, output_shift: int,
                   activation_min: int, activation_max: int) -> torch.Tensor:
    """The kernel's output for x (int8 or int16, normalized over its last dimension)."""
    n = x.shape[-1]
    rows = x.reshape(-1, n).to(torch.int64)
    s = rows.sum(dim=1)
    q = (rows * rows).sum(dim=1)
    m, lsh = _row_multiplier(n * q - s * s + int(epsilon))
    t = torch.bitwise_left_shift(n * rows - s[:, None], lsh[:, None])
    z = _qrdmulh(t, m[:, None])
    acc = _qrdmulh(z, gamma.to(torch.int64)[None, :]) + bias.to(torch.int64)[None, :]
    y = torch.bitwise_right_shift(acc + (1 << (output_shift - 1)), output_shift)
    return torch.clamp(y, activation_min, activation_max).to(x.dtype).reshape(x.shape)
