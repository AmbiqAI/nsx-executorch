# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# build() and `helia-torch compile --helia-aot`: a model heliaAOT fully covers becomes a plain heliaAOT module
# (no ExecuTorch runtime), any other model a PTE delegating what heliaAOT covers; module / delegate / off
# override that, and forcing a module on a model heliaAOT does not fully cover is an error.

import importlib.util
import json

import pytest
import torch

pytest.importorskip("helia_aot")

from nsx_cortex_m import quantize  # noqa: E402
from nsx_cortex_m.cli import main  # noqa: E402
from nsx_cortex_m.helia_aot import HeliaAotOptions, NotFullyDelegableError, build  # noqa: E402
from nsx_cortex_m.manifest import load_sidecar  # noqa: E402


class ConvRelu(torch.nn.Module):
    """Fully covered by heliaAOT."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


class ConvSigmoidConv(torch.nn.Module):
    """Mixed: aten.sigmoid has no AIR mapping, so it stays in ExecuTorch between two delegated convolutions."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(4, 8, 1)
        self.b = torch.nn.Conv2d(8, 8, 1)

    def forward(self, x):
        return self.b(torch.sigmoid(self.a(x)))


def _inputs():
    torch.manual_seed(0)
    return (torch.randn(1, 4, 6, 6).to(memory_format=torch.channels_last),)


def _quantized(model):
    torch.manual_seed(0)
    inputs = _inputs()
    return quantize(model.eval(), inputs, kernel_provider="ns", calibration_samples=[_inputs() for _ in range(4)])


def _options(out_dir):
    # Plain-C code generation, as in test_helia_aot.py.
    return HeliaAotOptions(output_dir=out_dir, platform="apollo4p_evb", capabilities=["MCU"])


@pytest.fixture()
def covered_pt2(tmp_path):
    path = tmp_path / "covered.pt2"
    torch.export.save(_quantized(ConvRelu()), path)
    return path


@pytest.fixture()
def mixed_pt2(tmp_path):
    path = tmp_path / "mixed.pt2"
    torch.export.save(_quantized(ConvSigmoidConv()), path)
    return path


# -- build() ----------------------------------------------------------------------------------------------------


def test_auto_builds_a_module_when_helia_aot_covers_the_model(tmp_path):
    built = build(_quantized(ConvRelu()), _inputs(), _options(tmp_path), name="net")
    assert built.mode == "module" and built.export_result is None and built.rejected == {}
    assert built.compiled.module_dir == tmp_path / "net"
    assert (tmp_path / "net" / "nsx-module.yaml").is_file()


def test_auto_delegates_a_mixed_model(tmp_path):
    built = build(_quantized(ConvSigmoidConv()), _inputs(), _options(tmp_path))
    assert built.mode == "delegate" and built.compiled is None
    assert "aten.sigmoid has no AIR mapping" in built.rejected.get("aten_sigmoid_default", "")
    ops = built.export_result.edge_ops
    assert any("sigmoid" in op for op in ops), ops


def test_auto_quantizes_a_float_model_once(tmp_path):
    built = build(ConvRelu().eval(), _inputs(), _options(tmp_path), calibration_samples=[_inputs()])
    assert built.mode == "module"


def test_forced_delegate_on_a_covered_model(tmp_path):
    built = build(_quantized(ConvRelu()), _inputs(), _options(tmp_path), mode="delegate")
    assert built.mode == "delegate" and built.rejected == {}


def test_forced_module_on_a_mixed_model_is_an_error(tmp_path):
    with pytest.raises(NotFullyDelegableError):
        build(_quantized(ConvSigmoidConv()), _inputs(), _options(tmp_path), mode="module")
    assert not any(tmp_path.iterdir()), "nothing is generated for a refused model"


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode must be one of"):
        build(_quantized(ConvRelu()), _inputs(), _options(tmp_path), mode="fastest")


# -- the CLI ----------------------------------------------------------------------------------------------------


def _compile(pt2, tmp_path, *extra):
    return main(["compile", str(pt2), "-o", str(tmp_path / "out" / "model.pte"), "--platform", "apollo4p_evb",
                 *extra])


def test_cli_default_writes_a_module_and_no_pte(covered_pt2, tmp_path, capsys):
    assert _compile(covered_pt2, tmp_path, "--name", "net") == 0
    out = tmp_path / "out"
    assert (out / "net" / "nsx-module.yaml").is_file()
    assert json.loads((out / "net" / "net_io.json").read_text())["module"] == "net"
    assert not (out / "model.pte").exists()
    assert "covers the whole model" in capsys.readouterr().out


def test_cli_default_delegates_a_mixed_model(mixed_pt2, tmp_path, capsys):
    assert _compile(mixed_pt2, tmp_path) == 0
    manifest = load_sidecar(tmp_path / "out" / "model.pte")
    assert manifest["requires_helia_aot"] and len(manifest["helia_aot_modules"]) == 2
    for entry in manifest["helia_aot_modules"]:
        assert (tmp_path / "out" / entry["module"] / "nsx-module.yaml").is_file()
    assert "aten_sigmoid_default" in capsys.readouterr().err


def test_cli_delegate_overrides_a_covered_model(covered_pt2, tmp_path):
    assert _compile(covered_pt2, tmp_path, "--helia-aot", "delegate") == 0
    manifest = load_sidecar(tmp_path / "out" / "model.pte")
    assert manifest["requires_helia_aot"] and len(manifest["helia_aot_modules"]) == 1


def test_cli_module_on_a_mixed_model_is_an_error(mixed_pt2, tmp_path):
    with pytest.raises(SystemExit, match="fully covers") as info:
        _compile(mixed_pt2, tmp_path, "--helia-aot", "module")
    assert "aten_sigmoid_default" in str(info.value)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_cli_off_lowers_without_helia_aot(covered_pt2, tmp_path):
    assert _compile(covered_pt2, tmp_path, "--helia-aot", "off") == 0
    manifest = load_sidecar(tmp_path / "out" / "model.pte")
    assert not manifest["requires_helia_aot"]
    assert any("conv" in op for op in manifest["operators"]["cortex_m"] + manifest["operators"]["cortex_m_ns"])


def test_cli_helia_aot_dir(covered_pt2, tmp_path):
    assert _compile(covered_pt2, tmp_path, "--name", "net", "--helia-aot-dir", str(tmp_path / "modules")) == 0
    assert (tmp_path / "modules" / "net" / "nsx-module.yaml").is_file()


def test_cli_arm_provider(covered_pt2, tmp_path):
    # auto quietly means off for the arm provider; an explicit heliaAOT mode is an error.
    assert _compile(covered_pt2, tmp_path, "--provider", "arm") == 0
    assert not load_sidecar(tmp_path / "out" / "model.pte")["requires_helia_aot"]
    with pytest.raises(SystemExit, match="needs --provider ns"):
        _compile(covered_pt2, tmp_path, "--provider", "arm", "--helia-aot", "delegate")


def test_cli_without_helia_aot_installed(covered_pt2, tmp_path, monkeypatch, capsys):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "helia_aot" else real(name, *a))
    assert _compile(covered_pt2, tmp_path) == 0
    assert not load_sidecar(tmp_path / "out" / "model.pte")["requires_helia_aot"]
    assert "heliaAOT is not installed" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="needs heliaAOT"):
        _compile(covered_pt2, tmp_path, "--helia-aot", "module")
