/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * cortex_m_ns::quantized_hardswish — int8 hardswish backed by the
 * ns-cmsis-nn arm_hard_swish_precise_s8 kernel.
 *
 * Kernel semantics (verified against the pinned ns-cmsis-nn source):
 *   x  = q_in - input_offset            (pass the raw input zero point)
 *   xr = clamp(x + relu_q3, 0, relu_q6)
 *   xr = nonneg_divide_by_pot(xr, prescale)
 *   y  = requantize(x * xr, output_multiplier, output_shift) + output_offset
 *   out = clamp(y, INT8_MIN, INT8_MAX)
 *
 * AOT parameters:
 *   relu_q3 = round(3 / input_scale), relu_q6 = round(6 / input_scale),
 *   output pair = quantize_multiplier(input_scale^2 / (6 * output_scale)
 *                                     * 2^prescale).
 * The AOT lowering emits prescale = 0 (matching the pinned unit tests) and
 * guards relu_q6 against int32 overflow of x * xr.
 */

#include "cortex_m_ns_ops_common.h"

namespace cortex_m_ns {
namespace native {

Tensor& quantized_hardswish_out(
    KernelRuntimeContext& context,
    const Tensor& input_int8,
    const int64_t input_zero_point,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
    const int64_t relu_q3,
    const int64_t relu_q6,
    const int64_t prescale,
    Tensor& out) {
  validate_cmsis_nn_tensor_requirements(
      input_int8,
      input_int8,
      out,
      ScalarType::Char,
      /*require_channels_last=*/false,
      /*require_same_sizes=*/true);
  validate_single_quant_params(
      input_zero_point, output_multiplier, output_shift, "Hardswish");

  // This kernel subtracts the input offset and adds the output offset, so
  // zero points are passed through unmodified (unlike the elementwise ops).
  arm_cmsis_nn_status status = arm_hard_swish_precise_s8(
      input_int8.data_ptr<int8_t>(),
      static_cast<int32_t>(input_zero_point),
      static_cast<int32_t>(output_zero_point),
      static_cast<int32_t>(output_multiplier),
      static_cast<int32_t>(output_shift),
      static_cast<int32_t>(relu_q3),
      static_cast<int32_t>(relu_q6),
      static_cast<int32_t>(prescale),
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(out.numel()));

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_hardswish_out: arm_hard_swish_precise_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m_ns
