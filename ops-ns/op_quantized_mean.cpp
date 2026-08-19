/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * cortex_m_ns::quantized_mean — int8 mean reduction backed by the
 * ns-cmsis-nn arm_mean_s8 kernel.
 *
 * Supported cases (enforced AOT by the lowering qualifiers and re-checked
 * here): contiguous (trivial dim-order) tensors of rank <= 4. The logical
 * shape is right-aligned into the 4D cmsis_nn_dims (n, h, w, c) and
 * `axis_mask` is the matching 4-element binary reduce mask. The element
 * count of the reduced dims is folded into (output_multiplier, output_shift)
 * ahead of time. Anything unsupported stays a portable aten.mean AOT.
 *
 * Offset conventions (verified against the pinned kernel source): the kernel
 * ADDS input_offset per element (pass the negated input zero point) and ADDS
 * out_offset after requantization (pass the raw output zero point).
 */

#include "cortex_m_ns_ops_common.h"

namespace cortex_m_ns {
namespace native {

Tensor& quantized_mean_out(
    KernelRuntimeContext& context,
    const Tensor& input_int8,
    const int64_t input_zero_point,
    const Int64ArrayRef axis_mask,
    const bool keepdim,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
    Tensor& out) {
  // keepdim only affects the logical output shape; the reduced data layout
  // is identical either way, so the value is not consumed here.
  (void)keepdim;
  ET_CHECK_MSG(
      input_int8.scalar_type() == ScalarType::Char &&
          out.scalar_type() == ScalarType::Char,
      "quantized_mean_out: tensors must be int8");
  ET_CHECK_MSG(
      input_int8.dim() <= 4,
      "quantized_mean_out: rank must be <= 4 [Value: %d]",
      static_cast<int>(input_int8.dim()));
  ET_CHECK_MSG(
      is_contiguous_dim_order_tensor(input_int8) &&
          is_contiguous_dim_order_tensor(out),
      "quantized_mean_out: tensors must be contiguous");
  ET_CHECK_MSG(
      axis_mask.size() == 4, "quantized_mean_out: axis_mask must have 4 entries");
  validate_single_quant_params(
      input_zero_point, output_multiplier, output_shift, "Mean");

  // Right-align the logical shape into (n, h, w, c).
  int32_t dims[4] = {1, 1, 1, 1};
  const int offset = 4 - static_cast<int>(input_int8.dim());
  for (int i = 0; i < input_int8.dim(); ++i) {
    dims[offset + i] = static_cast<int32_t>(input_int8.size(i));
  }

  cmsis_nn_dims input_dims = {dims[0], dims[1], dims[2], dims[3]};
  cmsis_nn_dims axis_dims = {
      static_cast<int32_t>(axis_mask[0]),
      static_cast<int32_t>(axis_mask[1]),
      static_cast<int32_t>(axis_mask[2]),
      static_cast<int32_t>(axis_mask[3])};
  cmsis_nn_dims output_dims = {
      axis_dims.n ? 1 : input_dims.n,
      axis_dims.h ? 1 : input_dims.h,
      axis_dims.w ? 1 : input_dims.w,
      axis_dims.c ? 1 : input_dims.c};

  const int64_t expected_out_numel = static_cast<int64_t>(output_dims.n) *
      output_dims.h * output_dims.w * output_dims.c;
  ET_CHECK_MSG(
      out.numel() == expected_out_numel,
      "quantized_mean_out: output numel mismatch [%" PRIi64 " vs %" PRIi64 "]",
      static_cast<int64_t>(out.numel()),
      expected_out_numel);

  arm_cmsis_nn_status status = arm_mean_s8(
      input_int8.data_ptr<int8_t>(),
      &input_dims,
      /*input_offset=*/-static_cast<int32_t>(input_zero_point),
      &axis_dims,
      out.mutable_data_ptr<int8_t>(),
      &output_dims,
      static_cast<int32_t>(output_zero_point),
      static_cast<int32_t>(output_multiplier),
      static_cast<int32_t>(output_shift));

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_mean_out: arm_mean_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m_ns
