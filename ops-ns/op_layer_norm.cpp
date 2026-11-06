/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * cortex_m_ns::layer_norm: float32 layer norm over the last dimension (ns-cmsis-nn arm_layer_norm_f32), for layer
 * norms the export leaves in float. When the ns-cmsis-nn build has no float32 kernels (ARM_NN_ENABLE_F32 off, the
 * default) or predates the layer-norm kernels, a plain C implementation runs instead, so a PTE that uses this
 * operator always runs.
 */

#include "cortex_m_ns_ops_common.h"

#include <cmath>

#if defined(__has_include)
#if __has_include(<Internal/arm_nn_layer_norm.h>) && defined(ARM_NN_ENABLE_F32)
#if ARM_NN_ENABLE_F32
#define NSX_EXECUTORCH_HAS_LAYER_NORM_F32 1
#endif
#endif
#endif

namespace cortex_m_ns {
namespace native {

namespace {
void layer_norm_plain_f32(
    int32_t rows,
    int32_t n,
    float eps,
    const float* x,
    const float* gamma,
    const float* beta,
    float* y) {
  for (int32_t r = 0; r < rows; r++, x += n, y += n) {
    float sum = 0.0f;
    for (int32_t c = 0; c < n; c++) {
      sum += x[c];
    }
    const float mean = sum / static_cast<float>(n);
    float sq = 0.0f;
    for (int32_t c = 0; c < n; c++) {
      const float d = x[c] - mean;
      sq += d * d;
    }
    const float inv_std = 1.0f / std::sqrt(sq / static_cast<float>(n) + eps);
    for (int32_t c = 0; c < n; c++) {
      const float t = (x[c] - mean) * inv_std;
      y[c] = t * (gamma ? gamma[c] : 1.0f) + (beta ? beta[c] : 0.0f);
    }
  }
}
} // namespace

Tensor& layer_norm_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const torch::executor::optional<Tensor>& weight,
    const torch::executor::optional<Tensor>& bias,
    const double eps,
    Tensor& out) {
  if (input.scalar_type() != ScalarType::Float || out.scalar_type() != ScalarType::Float || input.dim() < 1 ||
      input.numel() != out.numel() || !is_contiguous_dim_order_tensor(input) ||
      !is_contiguous_dim_order_tensor(out)) {
    ET_LOG(Error, "layer_norm_out: input and output must be contiguous float32 tensors of the same size");
    context.fail(Error::InvalidArgument);
    return out;
  }
  const int32_t n = static_cast<int32_t>(input.size(input.dim() - 1));
  for (const auto* t : {&weight, &bias}) {
    if (t->has_value() && (t->value().scalar_type() != ScalarType::Float || t->value().numel() != n)) {
      ET_LOG(Error, "layer_norm_out: weight and bias must be float32 with one value per element of a row");
      context.fail(Error::InvalidArgument);
      return out;
    }
  }
  const int32_t rows = n > 0 ? static_cast<int32_t>(input.numel() / n) : 0;
  const float* gamma = weight.has_value() ? weight.value().const_data_ptr<float>() : nullptr;
  const float* beta = bias.has_value() ? bias.value().const_data_ptr<float>() : nullptr;

#if defined(NSX_EXECUTORCH_HAS_LAYER_NORM_F32)
  const arm_cmsis_nn_status status = arm_layer_norm_f32(
      rows, n, static_cast<float>(eps), input.const_data_ptr<float>(), gamma, beta, out.mutable_data_ptr<float>());
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(Error, "layer_norm_out: arm_layer_norm_f32 failed with status [%d]", status);
    context.fail(Error::Internal);
  }
#else
  layer_norm_plain_f32(
      rows, n, static_cast<float>(eps), input.const_data_ptr<float>(), gamma, beta, out.mutable_data_ptr<float>());
#endif
  return out;
}

} // namespace native
} // namespace cortex_m_ns
