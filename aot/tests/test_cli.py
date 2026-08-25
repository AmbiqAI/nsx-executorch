# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# The CLI must take the same export() path as the Python API for both
# supported .pt2 flavors: a float32 program is quantized here (--calibrate),
# an already-INT8 program is lowered as-is and rejects --calibrate.

import pytest
import torch
from ns_tester import ramp_tensor

from nsx_cortex_m import quantize
from nsx_cortex_m.cli import main
from nsx_cortex_m.manifest import load_sidecar


class ConvReluModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def _example():
    return (ramp_tensor(-2, 2, (1, 2, 8, 8)),)


@pytest.fixture()
def float_pt2(tmp_path):
    torch.manual_seed(0)
    exported = torch.export.export(ConvReluModel().eval(), _example(), strict=True)
    path = tmp_path / "model_float.pt2"
    torch.export.save(exported, path)
    return path


@pytest.fixture()
def int8_pt2(tmp_path):
    torch.manual_seed(0)
    quantized = quantize(ConvReluModel().eval(), _example(), kernel_provider="arm")
    path = tmp_path / "model_int8.pt2"
    torch.export.save(quantized, path)
    return path


def _compile(pt2_path, out_path, *extra):
    return main(["compile", str(pt2_path), "-o", str(out_path), *extra])


def test_compile_float_pt2_quantizes_and_lowers(float_pt2, tmp_path):
    out = tmp_path / "float.pte"
    assert _compile(float_pt2, out, "--provider", "arm") == 0
    manifest = load_sidecar(out)
    assert any("cortex_m" in op for op in manifest["operators"]["cortex_m"])


def test_compile_int8_pt2_lowers_without_calibration(int8_pt2, tmp_path):
    out = tmp_path / "int8.pte"
    assert _compile(int8_pt2, out, "--provider", "arm") == 0
    manifest = load_sidecar(out)
    assert any("cortex_m" in op for op in manifest["operators"]["cortex_m"])


def test_compile_int8_and_float_pt2_lower_identically(float_pt2, int8_pt2, tmp_path):
    float_out = tmp_path / "from_float.pte"
    int8_out = tmp_path / "from_int8.pte"
    # Calibrate the float path with the example input, matching what
    # quantize() used for the int8 fixture, so the recipes coincide.
    assert _compile(float_pt2, float_out, "--provider", "arm", "--calibrate", "random:1") == 0
    assert _compile(int8_pt2, int8_out, "--provider", "arm") == 0
    float_ops = load_sidecar(float_out)["operators"]["cortex_m"]
    int8_ops = load_sidecar(int8_out)["operators"]["cortex_m"]
    assert float_ops == int8_ops


def test_compile_int8_pt2_rejects_calibrate_flag(int8_pt2, tmp_path):
    with pytest.raises(SystemExit, match="already PT2E-quantized"):
        _compile(int8_pt2, tmp_path / "x.pte", "--calibrate", "random:4")


def test_compile_int8_io_flag_serializes_int8_boundary(int8_pt2, tmp_path):
    out = tmp_path / "int8_io.pte"
    assert _compile(int8_pt2, out, "--provider", "arm", "--int8-io") == 0
    manifest = load_sidecar(out)
    assert manifest["inputs"][0]["dtype"] == "CHAR"  # ExecuTorch ScalarType name for int8
    assert manifest["outputs"][0]["dtype"] == "CHAR"


def test_export_int8_io_reports_qparams(int8_pt2):
    import torch as _torch

    from nsx_cortex_m import export

    exported = _torch.export.load(str(int8_pt2))
    result = export(exported, _example(), kernel_provider="arm", int8_io=True)
    (in_scale, in_zp, *_rest) = result.io_qparams["inputs"][0]
    assert in_scale > 0
    assert result.io_qparams["outputs"][0][4] == _torch.int8
