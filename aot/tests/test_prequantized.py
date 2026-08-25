# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# export() must accept an already-PT2E-quantized ExportedProgram (e.g. a
# loaded .pt2) and skip straight to kernel lowering: same lowered contract
# as when export() quantizes the float model itself, and calibration
# arguments are rejected since there is nothing left to calibrate.

import pytest
import torch
from ns_tester import ramp_tensor

from nsx_cortex_m import export, is_pt2e_quantized, quantize


class ConvReluModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def _example():
    return (ramp_tensor(-2, 2, (1, 2, 8, 8)),)


def _quantize_to_exported_program() -> torch.export.ExportedProgram:
    torch.manual_seed(0)
    return quantize(ConvReluModel().eval(), _example(), kernel_provider="arm")


def test_quantize_marks_program_quantized():
    program = _quantize_to_exported_program()
    assert is_pt2e_quantized(program)


def test_quantize_rejects_already_quantized_program():
    with pytest.raises(ValueError, match="already PT2E-quantized"):
        quantize(_quantize_to_exported_program(), _example())


@pytest.mark.parametrize("provider", ["arm", "ns"])
def test_prequantized_program_lowers_without_requantizing(provider):
    result = export(_quantize_to_exported_program(), _example(), kernel_provider=provider)
    cortex_ops = [op for op in result.edge_ops if op.startswith("cortex_m")]
    assert cortex_ops, f"expected cortex_m ops after lowering, got {result.edge_ops}"


def test_prequantized_matches_eager_float_path():
    torch.manual_seed(0)
    eager = export(ConvReluModel().eval(), _example(), kernel_provider="arm")
    prequantized = export(_quantize_to_exported_program(), _example(), kernel_provider="arm")
    assert prequantized.edge_ops == eager.edge_ops


def test_prequantized_rejects_calibration_samples():
    with pytest.raises(ValueError, match="already"):
        export(
            _quantize_to_exported_program(),
            _example(),
            calibration_samples=[_example()],
        )


def test_float_exported_program_is_still_quantized_here():
    torch.manual_seed(0)
    float_program = torch.export.export(ConvReluModel().eval(), _example(), strict=True)
    result = export(float_program, _example(), kernel_provider="arm")
    cortex_ops = [op for op in result.edge_ops if op.startswith("cortex_m")]
    assert cortex_ops, f"expected cortex_m ops after lowering, got {result.edge_ops}"
