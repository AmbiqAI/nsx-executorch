/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * cortex_m_ns::quantized_sub — int8 elementwise subtract backed by the
 * ns-cmsis-nn arm_elementwise_sub_s8 kernel. AOT parameter math mirrors the
 * stock cortex_m::quantized_add lowering (per-input rescale into a shared
 * 2*max(scale) domain with a fixed input left shift).
 *
 * Broadcasting: same conservative support as stock quantized_add/mul —
 * channel broadcast only. Unlike add/mul, subtraction is not commutative, so
 * the operands are never swapped; the wrapper instead advances whichever
 * operand (and the output) spans the full tensor.
 */

#include "cortex_m_ns_ops_common.h"

namespace cortex_m_ns {
namespace native {

Tensor& quantized_sub_out(
    KernelRuntimeContext& context,
    const Tensor& input1_int8,
    const int64_t input1_zero_point,
    const int64_t input1_multiplier,
    const int64_t input1_shift,
    const Tensor& input2_int8,
    const int64_t input2_zero_point,
    const int64_t input2_multiplier,
    const int64_t input2_shift,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
    const int64_t activation_min,
    const int64_t activation_max,
    Tensor& out) {
  const bool channel_broadcast =
      is_channel_broadcast(input1_int8, input2_int8);
  validate_cmsis_nn_tensor_requirements(
      input1_int8,
      input2_int8,
      out,
      ScalarType::Char,
      /*require_channels_last=*/channel_broadcast,
      /*require_same_sizes=*/!channel_broadcast);

  validate_quantization_params(
      input1_zero_point,
      input1_multiplier,
      input1_shift,
      input2_zero_point,
      input2_multiplier,
      input2_shift,
      output_zero_point,
      output_multiplier,
      output_shift,
      out);

  const int8_t* input1_ptr = input1_int8.data_ptr<int8_t>();
  const int8_t* input2_ptr = input2_int8.data_ptr<int8_t>();

  // CMSIS-NN offsets are added to the raw data while zero points are
  // subtracted when dequantizing; hence the negations below.
  const int32_t input1_offset = -static_cast<int32_t>(input1_zero_point);
  const int32_t input2_offset = -static_cast<int32_t>(input2_zero_point);

  int32_t subs_per_loop = 0;
  bool input1_advances = true;
  bool input2_advances = true;
  if (channel_broadcast) {
    subs_per_loop = static_cast<int32_t>(input1_int8.size(1));
    // The channels-only operand stays fixed while the full operand and the
    // output advance through the NHWC blocks.
    input1_advances = input1_int8.numel() != input1_int8.size(1);
    input2_advances = input2_int8.numel() != input2_int8.size(1);
  } else {
    subs_per_loop = static_cast<int32_t>(out.numel());
  }

  for (int32_t broadcast_offset = 0; broadcast_offset < out.numel();
       broadcast_offset += subs_per_loop) {
    arm_cmsis_nn_status status = arm_elementwise_sub_s8(
        input1_ptr + (input1_advances ? broadcast_offset : 0),
        input2_ptr + (input2_advances ? broadcast_offset : 0),
        input1_offset,
        static_cast<int32_t>(input1_multiplier),
        static_cast<int32_t>(input1_shift),
        input2_offset,
        static_cast<int32_t>(input2_multiplier),
        static_cast<int32_t>(input2_shift),
        kCmsisNnLeftShiftInt8,
        out.mutable_data_ptr<int8_t>() + broadcast_offset,
        static_cast<int32_t>(output_zero_point),
        static_cast<int32_t>(output_multiplier),
        static_cast<int32_t>(output_shift),
        static_cast<int32_t>(activation_min),
        static_cast<int32_t>(activation_max),
        subs_per_loop);

    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "quantized_sub_out: arm_elementwise_sub_s8 failed with status [%d]",
          status);
      context.fail(Error::Internal);
      return out;
    }
  }

  return out;
}

} // namespace native
} // namespace cortex_m_ns
