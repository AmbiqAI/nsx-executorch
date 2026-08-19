# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ns_tester import McuTestCase, NsCortexMTester, ramp_tensor

_Q = "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
_DQ = "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
_CM_Q = "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default"
_CM_DQ = "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default"


class SubModel(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 1,
        _Q: 3,
        _DQ: 3,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_ns_quantized_sub_default": 1,
        _CM_Q: 2,
        _CM_DQ: 1,
    }

    def forward(self, x, y):
        return x - y


test_cases = {
    "rank_1": McuTestCase(
        SubModel(),
        (ramp_tensor(-5, 5, (16,)), ramp_tensor(-2, 8, (16,))),
    ),
    "rank_2": McuTestCase(
        SubModel(),
        (ramp_tensor(-4, 4, (4, 8)), ramp_tensor(-2, 2, (4, 8))),
    ),
    "rank_4": McuTestCase(
        SubModel(),
        (ramp_tensor(-10, 10, (2, 3, 4, 4)), ramp_tensor(-1, 1, (2, 3, 4, 4))),
    ),
    "channel_broadcast_nhwc": McuTestCase(
        SubModel(),
        # Stock policy (CortexMAddMulCheck): channel broadcast is only
        # annotated for channels_last tensors; contiguous broadcast falls
        # back to portable aten.sub.
        (
            ramp_tensor(-4, 4, (2, 3, 4, 4)).to(memory_format=torch.channels_last),
            ramp_tensor(-1, 1, (1, 3, 1, 1)).to(memory_format=torch.channels_last),
        ),
    ),
}


@pytest.mark.parametrize("name", test_cases.keys())
def test_dialect_sub(name):
    case = test_cases[name]
    tester = NsCortexMTester(case.model, case.get_example_inputs())
    tester.test_dialect(
        case.model.ops_before_transforms, case.model.ops_after_transforms, qtol=1
    )
