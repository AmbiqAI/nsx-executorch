/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * cortex_m_ns::quantized_layer_norm: ns-cmsis-nn arm_layer_norm_s8 / arm_layer_norm_s16, layer norm over the last
 * dimension of an int8 or int16 tensor. gamma and bias are int32 [N] constants and epsilon / output_shift scalars,
 * folded at export time (aot/nsx_cortex_m/layer_norm_ns.py export_params()). The kernel's integer algorithm is
 * bit-exact with that file's layer_norm_int().
 *
 * Needs an ns-cmsis-nn with the layer-norm kernels (Internal/arm_nn_layer_norm.h); against an older one the operator
 * still builds but fails at runtime with NotSupported.
 */

#include "cortex_m_ns_ops_common.h"

#if defined(__has_include)
#if __has_include(<Internal/arm_nn_layer_norm.h>)
#define NSX_EXECUTORCH_HAS_LAYER_NORM 1
#endif
#endif

namespace cortex_m_ns {
namespace native {

Tensor& quantized_layer_norm_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& bias,
    const int64_t epsilon,
    const int64_t output_shift,
    const int64_t activation_min,
    const int64_t activation_max,
    Tensor& out) {
  const ScalarType dtype = input.scalar_type();
  if ((dtype != ScalarType::Char && dtype != ScalarType::Short) || out.scalar_type() != dtype) {
    ET_LOG(Error, "quantized_layer_norm_out: input and output must both be int8 or both int16");
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (input.dim() < 1 || input.numel() != out.numel() || !is_contiguous_dim_order_tensor(input) ||
      !is_contiguous_dim_order_tensor(out)) {
    ET_LOG(Error, "quantized_layer_norm_out: input and output must be contiguous and the same size");
    context.fail(Error::InvalidArgument);
    return out;
  }
  const int32_t row_size = static_cast<int32_t>(input.size(input.dim() - 1));
  if (gamma.scalar_type() != ScalarType::Int || bias.scalar_type() != ScalarType::Int ||
      gamma.numel() != row_size || bias.numel() != row_size) {
    ET_LOG(Error, "quantized_layer_norm_out: gamma and bias must be int32 with one value per element of a row");
    context.fail(Error::InvalidArgument);
    return out;
  }
  const int32_t rows = row_size > 0 ? static_cast<int32_t>(input.numel() / row_size) : 0;

#if defined(NSX_EXECUTORCH_HAS_LAYER_NORM)
  const cmsis_nn_layer_norm_params params = {
      epsilon,
      static_cast<int32_t>(output_shift),
      {static_cast<int32_t>(activation_min), static_cast<int32_t>(activation_max)}};
  const arm_cmsis_nn_status status = dtype == ScalarType::Char
      ? arm_layer_norm_s8(
            &params,
            rows,
            row_size,
            input.const_data_ptr<int8_t>(),
            gamma.const_data_ptr<int32_t>(),
            bias.const_data_ptr<int32_t>(),
            out.mutable_data_ptr<int8_t>())
      : arm_layer_norm_s16(
            &params,
            rows,
            row_size,
            input.const_data_ptr<int16_t>(),
            gamma.const_data_ptr<int32_t>(),
            bias.const_data_ptr<int32_t>(),
            out.mutable_data_ptr<int16_t>());
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(Error, "quantized_layer_norm_out: arm_layer_norm failed with status [%d]", status);
    context.fail(Error::Internal);
  }
#else
  (void)epsilon;
  (void)output_shift;
  (void)activation_min;
  (void)activation_max;
  (void)rows;
  ET_LOG(
      Error,
      "quantized_layer_norm_out: this ns-cmsis-nn has no arm_layer_norm kernels; build with one that does");
  context.fail(Error::NotSupported);
#endif
  return out;
}

} // namespace native
} // namespace cortex_m_ns
