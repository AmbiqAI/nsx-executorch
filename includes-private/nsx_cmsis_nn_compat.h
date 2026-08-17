#pragma once

#include <arm_nnfunctions.h>

#if !defined(NS_CMSIS_NN) || NS_CMSIS_NN_VERSION < 7002000
#error "nsx-executorch requires an NSX-capable ns-cmsis-nn release (>= 7.2.0)"
#endif

#include <cstdint>
#include <limits>

namespace nsx::executorch::cmsis_nn_compat {

inline std::int32_t aligned_size(std::int32_t size) {
  constexpr std::int32_t alignment = alignof(std::int32_t);
  if (size < 0 || size > std::numeric_limits<std::int32_t>::max() -
                           (alignment - 1)) {
    return -1;
  }
  return (size + alignment - 1) & -alignment;
}

inline std::int32_t combined_size(
    std::int32_t scratch_size, const cmsis_nn_dims* output_dims) {
  const std::int32_t weight_sum_size =
      arm_convolve_s8_get_weights_sum_size(output_dims);
  if (scratch_size < 0 || weight_sum_size < 0) {
    return -1;
  }
  if (weight_sum_size == 0) {
    return scratch_size;
  }
  const std::int32_t aligned_scratch = aligned_size(scratch_size);
  if (aligned_scratch < 0 ||
      aligned_scratch > std::numeric_limits<std::int32_t>::max() -
                            weight_sum_size) {
    return -1;
  }
  return aligned_scratch + weight_sum_size;
}

inline bool split_context(
    const cmsis_nn_context* combined,
    std::int32_t scratch_size,
    const cmsis_nn_dims* output_dims,
    cmsis_nn_context& scratch,
    cmsis_nn_context& weight_sum) {
  const std::int32_t weight_sum_size =
      arm_convolve_s8_get_weights_sum_size(output_dims);
  const std::int32_t total_size = combined_size(scratch_size, output_dims);
  if (total_size < 0) {
    return false;
  }
  if (total_size == 0) {
    scratch = {};
    weight_sum = {};
    return true;
  }
  if (combined == nullptr || combined->buf == nullptr ||
      combined->size < total_size) {
    return false;
  }

  scratch = {combined->buf, scratch_size};
  if (weight_sum_size > 0) {
    const std::int32_t aligned_scratch = aligned_size(scratch_size);
    auto* bytes = static_cast<std::uint8_t*>(combined->buf);
    weight_sum = {bytes + aligned_scratch, weight_sum_size};
  } else {
    weight_sum = {};
  }
  return true;
}

inline std::int32_t convolve_buffer_size(
    const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims,
    const cmsis_nn_dims* output_dims) {
  return combined_size(
      ::arm_convolve_wrapper_s8_get_buffer_size(
          conv_params, input_dims, filter_dims, output_dims),
      output_dims);
}

inline arm_cmsis_nn_status convolve(
    const cmsis_nn_context* context,
    const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims,
    const std::int8_t* input_data,
    const cmsis_nn_dims* filter_dims,
    const std::int8_t* filter_data,
    const cmsis_nn_dims* bias_dims,
    const std::int32_t* bias_data,
    const cmsis_nn_dims* output_dims,
    std::int8_t* output_data) {
  const std::int32_t scratch_size =
      ::arm_convolve_wrapper_s8_get_buffer_size(
          conv_params, input_dims, filter_dims, output_dims);
  cmsis_nn_context scratch{};
  cmsis_nn_context weight_sum{};
  if (!split_context(
          context, scratch_size, output_dims, scratch, weight_sum)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  if (weight_sum.size > 0) {
    const arm_cmsis_nn_status weight_status = ::arm_convolve_weight_sum(
        static_cast<std::int32_t*>(weight_sum.buf),
        filter_data,
        input_dims,
        filter_dims,
        output_dims,
        conv_params->input_offset,
        bias_data);
    if (weight_status != ARM_CMSIS_NN_SUCCESS) {
      return weight_status;
    }
  }
  return ::arm_convolve_wrapper_s8(
      &scratch,
      &weight_sum,
      conv_params,
      quant_params,
      input_dims,
      input_data,
      filter_dims,
      filter_data,
      bias_dims,
      bias_data,
      output_dims,
      output_data);
}

inline std::int32_t depthwise_buffer_size(
    const cmsis_nn_dw_conv_params* conv_params,
    const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims,
    const cmsis_nn_dims* output_dims) {
  return combined_size(
      ::arm_depthwise_conv_wrapper_s8_get_buffer_size(
          conv_params, input_dims, filter_dims, output_dims),
      output_dims);
}

inline arm_cmsis_nn_status depthwise(
    const cmsis_nn_context* context,
    const cmsis_nn_dw_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims,
    const std::int8_t* input_data,
    const cmsis_nn_dims* filter_dims,
    const std::int8_t* filter_data,
    const cmsis_nn_dims* bias_dims,
    const std::int32_t* bias_data,
    const cmsis_nn_dims* output_dims,
    std::int8_t* output_data) {
  const std::int32_t scratch_size =
      ::arm_depthwise_conv_wrapper_s8_get_buffer_size(
          conv_params, input_dims, filter_dims, output_dims);
  cmsis_nn_context scratch{};
  cmsis_nn_context weight_sum{};
  if (!split_context(
          context, scratch_size, output_dims, scratch, weight_sum)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  if (weight_sum.size > 0) {
    const arm_cmsis_nn_status weight_status =
        ::arm_depthwise_convolve_weight_sum(
            static_cast<std::int32_t*>(weight_sum.buf),
            static_cast<std::int8_t*>(scratch.buf),
            filter_data,
            conv_params,
            input_dims,
            filter_dims,
            output_dims,
            conv_params->input_offset,
            bias_data);
    if (weight_status != ARM_CMSIS_NN_SUCCESS) {
      return weight_status;
    }
  }
  return ::arm_depthwise_conv_wrapper_s8(
      &scratch,
      &weight_sum,
      conv_params,
      quant_params,
      input_dims,
      input_data,
      filter_dims,
      filter_data,
      bias_dims,
      bias_data,
      output_dims,
      output_data);
}

inline std::int32_t transpose_buffer_size(
    const cmsis_nn_transpose_conv_params* conv_params,
    const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims,
    const cmsis_nn_dims* output_dims) {
  return combined_size(
      ::arm_transpose_conv_s8_get_buffer_size(
          conv_params, input_dims, filter_dims, output_dims),
      output_dims);
}

inline arm_cmsis_nn_status transpose(
    const cmsis_nn_context* context,
    const cmsis_nn_context* output_context,
    const cmsis_nn_transpose_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims,
    const std::int8_t* input_data,
    const cmsis_nn_dims* filter_dims,
    const std::int8_t* filter_data,
    const cmsis_nn_dims* bias_dims,
    const std::int32_t* bias_data,
    const cmsis_nn_dims* output_dims,
    std::int8_t* output_data) {
  const std::int32_t scratch_size =
      ::arm_transpose_conv_s8_get_buffer_size(
          conv_params, input_dims, filter_dims, output_dims);
  cmsis_nn_context scratch{};
  cmsis_nn_context weight_sum{};
  if (!split_context(
          context, scratch_size, output_dims, scratch, weight_sum)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  if (weight_sum.size > 0) {
    const arm_cmsis_nn_status weight_status = ::arm_convolve_weight_sum(
        static_cast<std::int32_t*>(weight_sum.buf),
        filter_data,
        input_dims,
        filter_dims,
        output_dims,
        conv_params->input_offset,
        bias_data);
    if (weight_status != ARM_CMSIS_NN_SUCCESS) {
      return weight_status;
    }
  }
  return ::arm_transpose_conv_wrapper_s8(
      &scratch,
      &weight_sum,
      output_context,
      conv_params,
      quant_params,
      input_dims,
      input_data,
      filter_dims,
      filter_data,
      bias_dims,
      bias_data,
      output_dims,
      output_data);
}

}  // namespace nsx::executorch::cmsis_nn_compat

#define arm_convolve_wrapper_s8_get_buffer_size(...) \
  ::nsx::executorch::cmsis_nn_compat::convolve_buffer_size(__VA_ARGS__)
#define arm_convolve_wrapper_s8(...) \
  ::nsx::executorch::cmsis_nn_compat::convolve(__VA_ARGS__)
#define arm_depthwise_conv_wrapper_s8_get_buffer_size(...) \
  ::nsx::executorch::cmsis_nn_compat::depthwise_buffer_size(__VA_ARGS__)
#define arm_depthwise_conv_wrapper_s8(...) \
  ::nsx::executorch::cmsis_nn_compat::depthwise(__VA_ARGS__)
#define arm_transpose_conv_s8_get_buffer_size(...) \
  ::nsx::executorch::cmsis_nn_compat::transpose_buffer_size(__VA_ARGS__)
#define arm_transpose_conv_wrapper_s8(...) \
  ::nsx::executorch::cmsis_nn_compat::transpose(__VA_ARGS__)
