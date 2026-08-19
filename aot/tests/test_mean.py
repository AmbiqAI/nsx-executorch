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
_ATEN_MEAN = "executorch_exir_dialects_edge__ops_aten_mean_dim"
_NS_MEAN = "executorch_exir_dialects_edge__ops_cortex_m_ns_quantized_mean_default"


class MeanKeepdim(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


class MeanNoKeepdim(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(1,), keepdim=False)


class MeanGlobal(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(0, 1, 2, 3), keepdim=False)


class MeanRank5(torch.nn.Module):
    """Rank > 4 is outside what arm_mean_s8 supports; must stay portable."""

    def forward(self, x):
        return x.mean(dim=(1,), keepdim=False)


_BEFORE = {_ATEN_MEAN: 1, _Q: 2, _DQ: 2}
_AFTER = {_NS_MEAN: 1, _CM_Q: 1, _CM_DQ: 1}
# Qualifier failure: mean stays a portable aten op (correct but slow).
_AFTER_FALLBACK = {_ATEN_MEAN: 1, _CM_Q: 2, _CM_DQ: 2}

test_cases = {
    "keepdim_hw": (
        McuTestCase(MeanKeepdim(), (ramp_tensor(-4, 4, (2, 3, 4, 4)),)),
        _AFTER,
    ),
    "no_keepdim_rank3": (
        McuTestCase(MeanNoKeepdim(), (ramp_tensor(-4, 4, (2, 3, 4)),)),
        _AFTER,
    ),
    "global_rank4": (
        McuTestCase(MeanGlobal(), (ramp_tensor(-4, 4, (2, 3, 4, 4)),)),
        _AFTER,
    ),
    "rank5_fallback": (
        McuTestCase(MeanRank5(), (ramp_tensor(-4, 4, (2, 2, 2, 2, 2)),)),
        _AFTER_FALLBACK,
    ),
}


@pytest.mark.parametrize("name", test_cases.keys())
def test_dialect_mean(name):
    case, after = test_cases[name]
    tester = NsCortexMTester(case.model, case.get_example_inputs())
    tester.test_dialect(_BEFORE, after, qtol=1)
