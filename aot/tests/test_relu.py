# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Standalone relu / relu6 / hardtanh / clamp all lower to
# cortex_m_ns::quantized_relu (with requantization support).

import pytest
import torch
from ns_tester import McuTestCase, NsCortexMTester, ramp_tensor

_Q = "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
_DQ = "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
_CM_Q = "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default"
_CM_DQ = "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default"
_NS_RELU = "executorch_exir_dialects_edge__ops_cortex_m_ns_quantized_relu_default"


def _unary_expectations(aten_key: str):
    return (
        {aten_key: 1, _Q: 2, _DQ: 2},
        {_NS_RELU: 1, _CM_Q: 1, _CM_DQ: 1},
    )


class ReluModel(torch.nn.Module):
    aten_key = "executorch_exir_dialects_edge__ops_aten_relu_default"

    def __init__(self):
        super().__init__()
        self.act = torch.nn.ReLU()

    def forward(self, x):
        return self.act(x)


class Relu6Model(torch.nn.Module):
    aten_key = "executorch_exir_dialects_edge__ops_aten_hardtanh_default"

    def __init__(self):
        super().__init__()
        self.act = torch.nn.ReLU6()

    def forward(self, x):
        return self.act(x)


class HardtanhModel(torch.nn.Module):
    aten_key = "executorch_exir_dialects_edge__ops_aten_hardtanh_default"

    def __init__(self):
        super().__init__()
        self.act = torch.nn.Hardtanh(min_val=-2.0, max_val=3.0)

    def forward(self, x):
        return self.act(x)


class ClampModel(torch.nn.Module):
    aten_key = "executorch_exir_dialects_edge__ops_aten_clamp_default"

    def forward(self, x):
        return torch.clamp(x, -1.0, 2.0)


test_cases = {
    "relu": McuTestCase(ReluModel(), (ramp_tensor(-5, 5, (4, 8)),)),
    "relu_rank_4": McuTestCase(ReluModel(), (ramp_tensor(-5, 5, (2, 3, 4, 4)),)),
    "relu6": McuTestCase(Relu6Model(), (ramp_tensor(-8, 8, (4, 8)),)),
    "hardtanh": McuTestCase(HardtanhModel(), (ramp_tensor(-6, 6, (4, 8)),)),
    "clamp": McuTestCase(ClampModel(), (ramp_tensor(-4, 4, (4, 8)),)),
}


@pytest.mark.parametrize("name", test_cases.keys())
def test_dialect_relu_family(name):
    case = test_cases[name]
    before, after = _unary_expectations(case.model.aten_key)
    tester = NsCortexMTester(case.model, case.get_example_inputs())
    tester.test_dialect(before, after, qtol=1)
