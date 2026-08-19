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


class HardswishModel(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_hardswish_default": 1,
        _Q: 2,
        _DQ: 2,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_ns_quantized_hardswish_default": 1,
        _CM_Q: 1,
        _CM_DQ: 1,
    }

    def __init__(self):
        super().__init__()
        self.act = torch.nn.Hardswish()

    def forward(self, x):
        return self.act(x)


test_cases = {
    "rank_2": McuTestCase(HardswishModel(), (ramp_tensor(-6, 6, (8, 8)),)),
    "rank_4": McuTestCase(HardswishModel(), (ramp_tensor(-10, 10, (2, 3, 4, 4)),)),
    "narrow_range": McuTestCase(HardswishModel(), (ramp_tensor(-1, 1, (4, 8)),)),
}


@pytest.mark.parametrize("name", test_cases.keys())
def test_dialect_hardswish(name):
    case = test_cases[name]
    tester = NsCortexMTester(case.model, case.get_example_inputs())
    # qtol=2: hardswish multiplies two quantized terms (x * relu6(x+3)), so
    # the combined rounding error can reach 2 output LSBs on wide ranges.
    tester.test_dialect(
        case.model.ops_before_transforms, case.model.ops_after_transforms, qtol=2
    )
