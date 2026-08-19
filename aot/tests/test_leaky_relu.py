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


class LeakyReluModel(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_leaky_relu_default": 1,
        _Q: 2,
        _DQ: 2,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_ns_quantized_leaky_relu_default": 1,
        _CM_Q: 1,
        _CM_DQ: 1,
    }

    def __init__(self, alpha: float):
        super().__init__()
        self.act = torch.nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.act(x)


test_cases = {
    "alpha_pow2": McuTestCase(LeakyReluModel(0.125), (ramp_tensor(-5, 5, (4, 8)),)),
    "alpha_default": McuTestCase(LeakyReluModel(0.01), (ramp_tensor(-5, 5, (4, 8)),)),
    "alpha_large_rank4": McuTestCase(
        LeakyReluModel(0.3), (ramp_tensor(-10, 10, (2, 3, 4, 4)),)
    ),
}


@pytest.mark.parametrize("name", test_cases.keys())
def test_dialect_leaky_relu(name):
    case = test_cases[name]
    tester = NsCortexMTester(case.model, case.get_example_inputs())
    tester.test_dialect(
        case.model.ops_before_transforms, case.model.ops_after_transforms, qtol=1
    )
