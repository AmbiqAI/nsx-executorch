/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared helpers for the Ambiq ns-cmsis-nn additional operators
 * (`cortex_m_ns::` namespace). Reuses the stock Cortex-M validation helpers
 * from the unmodified ExecuTorch tree; this header adds only what the ns
 * kernels need on top.
 */

#pragma once

#include <executorch/backends/cortex_m/ops/cortex_m_ops_common.h>

namespace cortex_m_ns {
namespace native {

// CMSIS-NN input left shift used by the s8 elementwise add/sub kernels.
// Must match SHIFT_INT8 used by the AOT lowering.
constexpr int32_t kCmsisNnLeftShiftInt8 = 20;

// Sentinel requantization pair emitted by the AOT lowering when the requant
// is an exact identity (scale ratio 1.0): quantize_multiplier_aot(1.0).
constexpr int32_t kIdentityRequantMultiplier = 1 << 30;
constexpr int32_t kIdentityRequantShift = 1;

// True when the tensor uses the trivial (contiguous) dimension order.
inline bool is_contiguous_dim_order_tensor(const Tensor& tensor) {
  const auto dim_order = tensor.dim_order();
  for (size_t i = 0; i < dim_order.size(); ++i) {
    if (dim_order[i] != static_cast<executorch::aten::DimOrderType>(i)) {
      return false;
    }
  }
  return true;
}

} // namespace native
} // namespace cortex_m_ns
