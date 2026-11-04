/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * cortex_m_ns::quantized_conv2d_ws and cortex_m_ns::quantized_depthwise_conv2d_ws
 * -- the stock cortex_m::quantized_conv2d / quantized_depthwise_conv2d with
 * the ns-cmsis-nn per-output-channel weight sums precomputed at export time.
 *
 * The stock operators reach ns-cmsis-nn through the private adapter
 * (includes-private/nsx_cmsis_nn_compat.h), which has no per-operator state
 * and so recomputes the weight sums on every inference. heliaRT computes them
 * once at Prepare. Here the export pass (NsConvWeightSumPass) stores them in
 * the PTE as a constant int32 tensor:
 *
 *   weight_sum[j] = input_offset * sum(weights of output channel j) + bias[j]
 *
 * (32-bit wrapping), the layout arm_convolve_weight_sum() and
 * arm_depthwise_convolve_weight_sum() produce. ns-cmsis-nn kernels only read
 * the buffer and allow reusing it while filter, bias and input_offset are
 * unchanged, which holds for constants. `weight_sum_format` versions that
 * layout; a mismatch fails the call instead of computing wrong outputs.
 *
 * On builds without MVE, ns-cmsis-nn uses no weight sums
 * (arm_convolve_s8_get_weights_sum_size() == 0) and the tensor is ignored.
 * Validation mirrors the stock operators.
 */

#include "cortex_m_ns_ops_common.h"

#define NSX_EXECUTORCH_CMSIS_NN_COMPAT_NO_MACROS
#include "nsx_cmsis_nn_compat.h"

namespace cortex_m_ns {
namespace native {

namespace {

constexpr int64_t kConvDim = 4;

// Must match WEIGHT_SUM_FORMAT in aot/nsx_cortex_m/weight_sums_ns.py.
constexpr int64_t kWeightSumFormat = 1;

bool validate_common(
    KernelRuntimeContext& context,
    const char* op,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Tensor& output,
    const Int64ArrayRef& stride,
    const Int64ArrayRef& padding,
    const Int64ArrayRef& dilation,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const Tensor& weight_sum,
    const int64_t weight_sum_format) {
  if (weight_sum_format != kWeightSumFormat) {
    ET_LOG(
        Error,
        "%s: weight_sum_format %d is not the runtime's format %d; "
        "re-export the model with a matching helia-torch",
        op,
        static_cast<int>(weight_sum_format),
        static_cast<int>(kWeightSumFormat));
    context.fail(Error::InvalidProgram);
    return false;
  }
  if (input.dim() != kConvDim || weight.dim() != kConvDim ||
      output.dim() != kConvDim) {
    ET_LOG(Error, "%s: tensors must be 4-D", op);
    context.fail(Error::InvalidArgument);
    return false;
  }
  if (input.scalar_type() != ScalarType::Char ||
      output.scalar_type() != ScalarType::Char ||
      weight.scalar_type() != ScalarType::Char) {
    ET_LOG(Error, "%s: input, weight and output must be int8", op);
    context.fail(Error::InvalidArgument);
    return false;
  }
  if (bias.has_value() && bias.value().scalar_type() != ScalarType::Int) {
    ET_LOG(Error, "%s: bias must be int32", op);
    context.fail(Error::InvalidArgument);
    return false;
  }
  if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) {
    ET_LOG(Error, "%s: stride/padding/dilation must have length 2", op);
    context.fail(Error::InvalidArgument);
    return false;
  }
  const int64_t out_channels = output.size(1);
  if (requantize_multipliers.size(0) != out_channels ||
      requantize_shifts.size(0) != out_channels) {
    ET_LOG(Error, "%s: per-channel params size != out_ch (%zd)", op, out_channels);
    context.fail(Error::InvalidArgument);
    return false;
  }
  if (weight_sum.scalar_type() != ScalarType::Int || weight_sum.dim() != 1 ||
      weight_sum.size(0) != out_channels) {
    ET_LOG(Error, "%s: weight_sum must be int32 [out_ch=%zd]", op, out_channels);
    context.fail(Error::InvalidArgument);
    return false;
  }
  return true;
}

// The weight-sum context for the kernels: the precomputed constants on MVE
// builds, empty where ns-cmsis-nn takes none.
bool weight_sum_context(
    KernelRuntimeContext& context,
    const char* op,
    const Tensor& weight_sum,
    const cmsis_nn_dims& output_dims,
    cmsis_nn_context& ctx) {
  const int32_t size = arm_convolve_s8_get_weights_sum_size(&output_dims);
  if (size < 0) {
    ET_LOG(Error, "%s: weight-sum size query failed", op);
    context.fail(Error::Internal);
    return false;
  }
  ctx.size = size;
  // Read only: the kernels never write the weight-sum buffer.
  ctx.buf = size > 0
      ? const_cast<int32_t*>(weight_sum.const_data_ptr<int32_t>())
      : nullptr;
  return true;
}

bool allocate_scratch(
    KernelRuntimeContext& context,
    const char* op,
    const int32_t bytes,
    cmsis_nn_context& ctx) {
  ctx.buf = nullptr;
  ctx.size = 0;
  if (bytes < 0) {
    ET_LOG(Error, "%s: CMSIS-NN buffer size calculation failed", op);
    context.fail(Error::Internal);
    return false;
  }
  if (bytes == 0) {
    return true;
  }
  auto buffer_or_error =
      context.allocate_temp(static_cast<size_t>(bytes), kCortexMMveAlignment);
  if (!buffer_or_error.ok()) {
    ET_LOG(
        Error,
        "%s: failed to allocate scratch buffer (%d bytes, error %d)",
        op,
        static_cast<int>(bytes),
        static_cast<int>(buffer_or_error.error()));
    context.fail(buffer_or_error.error());
    return false;
  }
  ctx.buf = buffer_or_error.get();
  ctx.size = bytes;
  return true;
}

} // namespace

Tensor& quantized_conv2d_ws_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    const Tensor& weight_sum,
    const int64_t weight_sum_format,
    Tensor& out) {
  constexpr const char* kOp = "quantized_conv2d_ws_out";
  if (!validate_common(
          context, kOp, input, weight, bias, out, stride, padding, dilation,
          requantize_multipliers, requantize_shifts, weight_sum,
          weight_sum_format)) {
    return out;
  }
  // Stock rule: dim_order is ambiguous for a single channel.
  if (input.size(1) > 1 && !is_channels_last_tensor(input)) {
    ET_LOG(Error, "%s: input must have channels_last dim_order (NHWC)", kOp);
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (out.size(1) > 1 && !is_channels_last_tensor(out)) {
    ET_LOG(Error, "%s: output must have channels_last dim_order (NHWC)", kOp);
    context.fail(Error::InvalidArgument);
    return out;
  }

  // Weight is OHWI after the stock pass: [C_OUT, H, W, C_IN].
  const cmsis_nn_dims input_dims{
      static_cast<int32_t>(input.size(0)),
      static_cast<int32_t>(input.size(2)),
      static_cast<int32_t>(input.size(3)),
      static_cast<int32_t>(input.size(1))};
  const cmsis_nn_dims filter_dims{
      static_cast<int32_t>(weight.size(0)),
      static_cast<int32_t>(weight.size(1)),
      static_cast<int32_t>(weight.size(2)),
      static_cast<int32_t>(weight.size(3))};
  const cmsis_nn_dims output_dims{
      static_cast<int32_t>(out.size(0)),
      static_cast<int32_t>(out.size(2)),
      static_cast<int32_t>(out.size(3)),
      static_cast<int32_t>(out.size(1))};
  const cmsis_nn_dims bias_dims{1, 1, 1, output_dims.c};

  cmsis_nn_conv_params conv_params;
  conv_params.input_offset = static_cast<int32_t>(input_offset);
  conv_params.output_offset = static_cast<int32_t>(output_offset);
  conv_params.stride.h = static_cast<int32_t>(stride[0]);
  conv_params.stride.w = static_cast<int32_t>(stride[1]);
  conv_params.padding.h = static_cast<int32_t>(padding[0]);
  conv_params.padding.w = static_cast<int32_t>(padding[1]);
  conv_params.dilation.h = static_cast<int32_t>(dilation[0]);
  conv_params.dilation.w = static_cast<int32_t>(dilation[1]);
  conv_params.activation.min = static_cast<int32_t>(activation_min);
  conv_params.activation.max = static_cast<int32_t>(activation_max);

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = requantize_multipliers.data_ptr<int32_t>();
  quant_params.shift = requantize_shifts.data_ptr<int32_t>();

  cmsis_nn_context weight_sum_ctx;
  if (!weight_sum_context(context, kOp, weight_sum, output_dims, weight_sum_ctx)) {
    return out;
  }
  cmsis_nn_context scratch;
  if (!allocate_scratch(
          context,
          kOp,
          arm_convolve_wrapper_s8_get_buffer_size(
              &conv_params, &input_dims, &filter_dims, &output_dims),
          scratch)) {
    return out;
  }

  const arm_cmsis_nn_status status = arm_convolve_wrapper_s8(
      &scratch,
      &weight_sum_ctx,
      &conv_params,
      &quant_params,
      &input_dims,
      input.const_data_ptr<int8_t>(),
      &filter_dims,
      weight.const_data_ptr<int8_t>(),
      &bias_dims,
      bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr,
      &output_dims,
      out.mutable_data_ptr<int8_t>());
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(Error, "%s: arm_convolve_wrapper_s8 failed with status %d", kOp, status);
    context.fail(Error::Internal);
  }
  return out;
}

Tensor& quantized_depthwise_conv2d_ws_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    const int64_t depth_multiplier,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    const Tensor& weight_sum,
    const int64_t weight_sum_format,
    Tensor& out) {
  constexpr const char* kOp = "quantized_depthwise_conv2d_ws_out";
  if (!validate_common(
          context, kOp, input, weight, bias, out, stride, padding, dilation,
          requantize_multipliers, requantize_shifts, weight_sum,
          weight_sum_format)) {
    return out;
  }
  // CMSIS-NN depthwise convolution supports batch 1 only; weight is IHWO
  // [1, H, W, C_OUT] after the stock pass.
  if (input.size(0) != 1 || weight.size(0) != 1 ||
      weight.size(3) != out.size(1)) {
    ET_LOG(Error, "%s: need batch 1 and an IHWO [1, H, W, out_ch] weight", kOp);
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (!is_channels_last_tensor(input) || !is_channels_last_tensor(out)) {
    ET_LOG(Error, "%s: input and output must be channels_last", kOp);
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (out.size(1) != input.size(1) * depth_multiplier) {
    ET_LOG(Error, "%s: out_ch != in_ch * depth_multiplier", kOp);
    context.fail(Error::InvalidArgument);
    return out;
  }

  const cmsis_nn_dims input_dims{
      static_cast<int32_t>(input.size(0)),
      static_cast<int32_t>(input.size(2)),
      static_cast<int32_t>(input.size(3)),
      static_cast<int32_t>(input.size(1))};
  const cmsis_nn_dims filter_dims{
      1,
      static_cast<int32_t>(weight.size(1)),
      static_cast<int32_t>(weight.size(2)),
      static_cast<int32_t>(out.size(1))};
  const cmsis_nn_dims output_dims{
      static_cast<int32_t>(out.size(0)),
      static_cast<int32_t>(out.size(2)),
      static_cast<int32_t>(out.size(3)),
      static_cast<int32_t>(out.size(1))};
  const cmsis_nn_dims bias_dims{1, 1, 1, output_dims.c};

  cmsis_nn_dw_conv_params dw_conv_params;
  dw_conv_params.input_offset = static_cast<int32_t>(input_offset);
  dw_conv_params.output_offset = static_cast<int32_t>(output_offset);
  dw_conv_params.ch_mult = static_cast<int32_t>(depth_multiplier);
  dw_conv_params.stride.h = static_cast<int32_t>(stride[0]);
  dw_conv_params.stride.w = static_cast<int32_t>(stride[1]);
  dw_conv_params.padding.h = static_cast<int32_t>(padding[0]);
  dw_conv_params.padding.w = static_cast<int32_t>(padding[1]);
  dw_conv_params.dilation.h = static_cast<int32_t>(dilation[0]);
  dw_conv_params.dilation.w = static_cast<int32_t>(dilation[1]);
  dw_conv_params.activation.min = static_cast<int32_t>(activation_min);
  dw_conv_params.activation.max = static_cast<int32_t>(activation_max);

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = requantize_multipliers.data_ptr<int32_t>();
  quant_params.shift = requantize_shifts.data_ptr<int32_t>();

  cmsis_nn_context weight_sum_ctx;
  if (!weight_sum_context(context, kOp, weight_sum, output_dims, weight_sum_ctx)) {
    return out;
  }
  // Sized for the 3x3 direct entries as well as the wrapper.
  cmsis_nn_context scratch;
  if (!allocate_scratch(
          context,
          kOp,
          nsx::executorch::cmsis_nn_compat::depthwise_scratch_size(
              &dw_conv_params, &input_dims, &filter_dims, &output_dims),
          scratch)) {
    return out;
  }

  const arm_cmsis_nn_status status =
      nsx::executorch::cmsis_nn_compat::depthwise_with_weight_sum(
          &scratch,
          &weight_sum_ctx,
          &dw_conv_params,
          &quant_params,
          &input_dims,
          input.const_data_ptr<int8_t>(),
          &filter_dims,
          weight.const_data_ptr<int8_t>(),
          &bias_dims,
          bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr,
          &output_dims,
          out.mutable_data_ptr<int8_t>());
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(Error, "%s: depthwise kernel failed with status %d", kOp, status);
    context.fail(Error::Internal);
  }
  return out;
}

} // namespace native
} // namespace cortex_m_ns
