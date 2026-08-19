# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# The arm kernel provider must be completely unaffected by the NS work:
# exporting the same Tier 1 models with kernel_provider="arm" must yield
# ZERO cortex_m_ns ops (they stay portable aten or stock-decomposed ops).

import pytest
import torch
from ns_tester import ramp_tensor

from nsx_cortex_m import export


class SubModel(torch.nn.Module):
    def forward(self, x, y):
        return x - y


class MeanModel(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


class MulModel(torch.nn.Module):
    def forward(self, x):
        return x * x


arm_cases = {
    "sub": (SubModel(), (ramp_tensor(-4, 4, (4, 8)), ramp_tensor(-2, 2, (4, 8)))),
    "hardswish": (torch.nn.Hardswish(), (ramp_tensor(-6, 6, (8, 8)),)),
    "relu": (torch.nn.ReLU(), (ramp_tensor(-5, 5, (4, 8)),)),
    "relu6": (torch.nn.ReLU6(), (ramp_tensor(-8, 8, (4, 8)),)),
    "mean": (MeanModel(), (ramp_tensor(-4, 4, (2, 3, 4, 4)),)),
    "leaky_relu": (torch.nn.LeakyReLU(0.125), (ramp_tensor(-5, 5, (4, 8)),)),
}


@pytest.mark.parametrize("name", arm_cases.keys())
def test_arm_export_has_no_ns_ops(name):
    model, inputs = arm_cases[name]
    result = export(model, inputs, kernel_provider="arm")
    ns_ops = [op for op in result.edge_ops if "cortex_m_ns" in op]
    assert not ns_ops, f"arm export must not contain cortex_m_ns ops, got {ns_ops}"


def test_arm_export_stock_lowering_intact():
    """Sanity: the arm path still performs the stock cortex_m lowering."""
    result = export(MulModel(), (ramp_tensor(-5, 5, (10,)),), kernel_provider="arm")
    assert result.edge_ops.get("cortex_m::quantized_mul") == 1, result.edge_ops
    assert not [op for op in result.edge_ops if "cortex_m_ns" in op]


def test_ns_export_produces_ns_ops():
    """Sanity inverse: the ns path lowers to cortex_m_ns ops."""
    result = export(
        SubModel(),
        (ramp_tensor(-4, 4, (4, 8)), ramp_tensor(-2, 2, (4, 8))),
        kernel_provider="ns",
    )
    assert result.edge_ops.get("cortex_m_ns::quantized_sub") == 1, result.edge_ops
    assert not result.portable_fallback_ops


def test_ns_fallback_report():
    """Qualifier failures surface in the portable fallback report."""

    class Rank5Mean(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(1,), keepdim=False)

    result = export(
        Rank5Mean(), (ramp_tensor(-4, 4, (2, 2, 2, 2, 2)),), kernel_provider="ns"
    )
    assert result.edge_ops.get("aten::mean.dim") == 1, result.edge_ops
    assert result.portable_fallback_ops == ["aten::mean.out"]
    assert result.portable_select_ops_list == "aten::mean.out"
