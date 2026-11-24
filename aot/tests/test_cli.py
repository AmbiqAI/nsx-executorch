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


def test_compile_float_pt2_with_int8_io_quantizes_then_strips_boundary(float_pt2, tmp_path):
    """The quantize-here-then-strip flow: a float .pt2 with --int8-io must be
    PT2E-quantized inside export() and still serialize an int8 boundary."""
    out = tmp_path / "float_int8_io.pte"
    assert _compile(float_pt2, out, "--provider", "arm", "--int8-io") == 0
    manifest = load_sidecar(out)
    assert manifest["inputs"][0]["dtype"] == "CHAR"  # ExecuTorch ScalarType name for int8
    assert manifest["outputs"][0]["dtype"] == "CHAR"
    assert any("cortex_m" in op for op in manifest["operators"]["cortex_m"])


def test_export_int8_io_from_eager_float_model_reports_qparams():
    import torch as _torch

    from nsx_cortex_m import export

    _torch.manual_seed(0)
    result = export(
        ConvReluModel().eval(), _example(), kernel_provider="arm", int8_io=True
    )
    assert result.io_qparams["inputs"][0][0] > 0  # scale
    assert result.io_qparams["outputs"][0][4] == _torch.int8


def test_compile_rejects_non_tensor_positional_inputs(tmp_path):
    import torch as _torch

    class ScaledAdd(_torch.nn.Module):
        def forward(self, x, factor: int):
            return x + factor

    exported = _torch.export.export(
        ScaledAdd().eval(), (ramp_tensor(-2, 2, (4, 8)), 3), strict=True
    )
    path = tmp_path / "non_tensor.pt2"
    _torch.export.save(exported, path)
    with pytest.raises(SystemExit, match="non-tensor inputs"):
        _compile(path, tmp_path / "x.pte")


# --no-quantize: a float model lowers in its own dtype on ExecuTorch's portable kernels.


class FloatModel(torch.nn.Module):
    """Linear and hardswish, which the int8 edge config keeps whole and which have no portable kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.linear = torch.nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        return torch.softmax(self.linear(torch.nn.functional.hardswish(self.conv(x)).flatten(1)), dim=-1)


def _portable_kernels() -> set[str]:
    """The operators ExecuTorch's portable kernel library implements, from its functions.yaml."""
    import re
    from importlib.resources import files

    text = (files("executorch") / "kernels" / "portable" / "functions.yaml").read_text()
    aten = {f"aten::{name}" for name in re.findall(r"^- op: (\S+)$", text, flags=re.MULTILINE)}
    # Operators outside aten are listed with their schema and their namespace.
    return aten | set(re.findall(r"^- func: (\w+::[\w.]+)\(", text, flags=re.MULTILINE))


def _float_pt2(tmp_path, dtype):
    torch.manual_seed(0)
    model = FloatModel().eval().to(dtype)
    example = (ramp_tensor(-2, 2, (1, 2, 8, 8)).to(dtype),)
    path = tmp_path / f"model_{str(dtype).split('.')[-1]}.pt2"
    torch.export.save(torch.export.export(model, example, strict=True), path)
    return path, model, example


@pytest.mark.parametrize(
    ("dtype", "scalar_type", "tolerance"),
    [(torch.float32, "FLOAT", 1e-6), (torch.float16, "HALF", 1e-2), (torch.bfloat16, "BFLOAT16", 5e-2)],
)
def test_compile_no_quantize_lowers_a_float_model_on_portable_kernels(tmp_path, dtype, scalar_type, tolerance):
    from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer

    path, model, example = _float_pt2(tmp_path, dtype)
    out = tmp_path / "float.pte"
    assert _compile(path, out, "--no-quantize") == 0

    manifest = load_sidecar(out)
    assert manifest["operators"]["cortex_m"] == manifest["operators"]["cortex_m_ns"] == []
    assert manifest["helia_aot_modules"] == []
    assert [tensor["dtype"] for tensor in manifest["inputs"] + manifest["outputs"]] == [scalar_type] * 2
    # Every operator must have a portable kernel, or the firmware cannot run the PTE.
    assert set(manifest["operators"]["portable"]) <= _portable_kernels()

    (output,) = _load_for_executorch_from_buffer(out.read_bytes()).forward(list(example))
    assert output.dtype == dtype
    torch.testing.assert_close(output.float(), model(*example).float(), atol=tolerance, rtol=0)


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (("--calibrate", "random:2"), "--calibrate"),
        (("--int8-io",), "--int8-io"),
        (("--helia-aot", "module"), "--helia-aot module"),
        (("--helia-aot", "delegate"), "--helia-aot delegate"),
        (("--test-case",), "--test-case"),
        (("--name", "kws"), "--name"),
        (("--debug-program",), "--debug-program"),
    ],
)
def test_compile_no_quantize_refuses_what_needs_an_int8_model(float_pt2, tmp_path, flags, named):
    with pytest.raises(SystemExit, match=f"{named} needs an int8 model"):
        _compile(float_pt2, tmp_path / "x.pte", "--no-quantize", *flags)


def test_compile_no_quantize_lowers_an_int8_model_as_usual(int8_pt2, tmp_path):
    out = tmp_path / "int8.pte"
    assert _compile(int8_pt2, out, "--provider", "arm", "--no-quantize") == 0
    assert load_sidecar(out)["operators"]["cortex_m"]


def test_export_without_quantization_refuses_int8_options():
    from nsx_cortex_m import export

    with pytest.raises(ValueError, match="int8_io: only for an int8 program"):
        export(ConvReluModel().eval(), _example(), kernel_provider="ns", quantize_model=False, int8_io=True)
