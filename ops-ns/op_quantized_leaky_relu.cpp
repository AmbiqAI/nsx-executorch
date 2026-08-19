/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * cortex_m_ns::quantized_leaky_relu — int8 leaky relu backed by the
 * ns-cmsis-nn arm_leaky_relu_s8 kernel.
 *
 * Kernel semantics: x = q_in - input_offset; x >= 0 requantizes with the
 * identity pair (input_scale / output_scale), x < 0 with the alpha pair
 * (input_scale * alpha / output_scale); adds the output offset and clamps to
 * int8. Zero points are passed through unmodified.
 */

#include "cortex_m_ns_ops_common.h"

namespace cortex_m_ns {
namespace native {

Tensor& quantized_leaky_relu_out(
    KernelRuntimeContext& context,
    const Tensor& input_int8,
    const int64_t input_zero_point,
    const int64_t output_zero_point,
    const int64_t alpha_multiplier,
    const int64_t alpha_shift,
    const int64_t identity_multiplier,
    const int64_t identity_shift,
    Tensor& out) {
  validate_cmsis_nn_tensor_requirements(
      input_int8,
      input_int8,
      out,
      ScalarType::Char,
      /*require_channels_last=*/false,
      /*require_same_sizes=*/true);
  validate_single_quant_params(
      input_zero_point, alpha_multiplier, alpha_shift, "LeakyRelu alpha");
  validate_single_quant_params(
      output_zero_point,
      identity_multiplier,
      identity_shift,
      "LeakyRelu identity");

  arm_cmsis_nn_status status = arm_leaky_relu_s8(
      input_int8.data_ptr<int8_t>(),
      static_cast<int32_t>(input_zero_point),
      static_cast<int32_t>(output_zero_point),
      static_cast<int32_t>(alpha_multiplier),
      static_cast<int32_t>(alpha_shift),
      static_cast<int32_t>(identity_multiplier),
      static_cast<int32_t>(identity_shift),
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(out.numel()));

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_leaky_relu_out: arm_leaky_relu_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m_ns
