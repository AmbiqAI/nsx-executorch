/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * cortex_m_ns::quantized_relu — int8 clamp-style activation (relu, relu6,
 * hardtanh, clamp with scalar bounds) backed by ns-cmsis-nn.
 *
 * Two execution paths:
 *  - Identity requantization (same qparams on input and output, sentinel
 *    multiplier/shift emitted AOT): arm_clamp_s8, a pure clamp.
 *  - Otherwise: arm_relu_generic_s8, which subtracts the input offset,
 *    requantizes with (output_multiplier, output_shift), adds the output
 *    offset and clamps to [activation_min, activation_max] in the output
 *    quantized domain.
 */

#include "cortex_m_ns_ops_common.h"

namespace cortex_m_ns {
namespace native {

Tensor& quantized_relu_out(
    KernelRuntimeContext& context,
    const Tensor& input_int8,
    const int64_t input_zero_point,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
    const int64_t activation_min,
    const int64_t activation_max,
    Tensor& out) {
  validate_cmsis_nn_tensor_requirements(
      input_int8,
      input_int8,
      out,
      ScalarType::Char,
      /*require_channels_last=*/false,
      /*require_same_sizes=*/true);
  validate_single_quant_params(
      input_zero_point, output_multiplier, output_shift, "Relu");

  const int32_t size = static_cast<int32_t>(out.numel());
  const bool identity_requant = input_zero_point == output_zero_point &&
      output_multiplier == kIdentityRequantMultiplier &&
      output_shift == kIdentityRequantShift;

  if (identity_requant) {
    arm_cmsis_nn_status status = arm_clamp_s8(
        input_int8.data_ptr<int8_t>(),
        static_cast<int8_t>(activation_min),
        static_cast<int8_t>(activation_max),
        out.mutable_data_ptr<int8_t>(),
        size);
    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "quantized_relu_out: arm_clamp_s8 failed with status [%d]",
          status);
      context.fail(Error::Internal);
    }
    return out;
  }

  // This kernel subtracts the input offset and adds the output offset, so
  // zero points are passed through unmodified (unlike the elementwise ops).
  arm_cmsis_nn_status status = arm_relu_generic_s8(
      input_int8.data_ptr<int8_t>(),
      static_cast<int32_t>(input_zero_point),
      static_cast<int32_t>(output_zero_point),
      static_cast<int32_t>(output_multiplier),
      static_cast<int32_t>(output_shift),
      static_cast<int32_t>(activation_min),
      static_cast<int32_t>(activation_max),
      out.mutable_data_ptr<int8_t>(),
      size);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_relu_out: arm_relu_generic_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m_ns
