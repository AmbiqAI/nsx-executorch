# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Layer norm on ns-cmsis-nn's arm_layer_norm kernels: the NS lowering, the export-time parameters, and the
# integer reference (layer_norm_ns.py) against ns-cmsis-nn's own reference and C kernels.
#
# The cross-checks need NSX_EXECUTORCH_TEST_NS_CMSIS_NN_ROOT pointing at an ns-cmsis-nn checkout that has the
# layer-norm kernels (Include/Internal/arm_nn_layer_norm.h) and a host C compiler; they skip otherwise.

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig

from nsx_cortex_m import layer_norm_ns as L
from nsx_cortex_m.export import export, quantize
from nsx_cortex_m.manifest import build_manifest

TARGET = CortexMTargetConfig(cpu=CortexM.M55)
QLN = "cortex_m_ns::quantized_layer_norm"
FLN = "cortex_m_ns::layer_norm"
ATEN_LN = "aten::native_layer_norm"


class Block(torch.nn.Module):
    """A transformer-style projection: linear -> layer norm -> linear."""

    def __init__(self, d_in=16, d=32, d_out=8, affine=True):
        super().__init__()
        self.inp = torch.nn.Linear(d_in, d)
        self.ln = torch.nn.LayerNorm(d, elementwise_affine=affine)
        self.out = torch.nn.Linear(d, d_out)
        if affine:
            with torch.no_grad():
                self.ln.weight.normal_(1.0, 0.3)
                self.ln.weight[::5] *= -1
                self.ln.bias.normal_(0.0, 0.2)

    def forward(self, x):
        return self.out(self.ln(self.inp(x)))


def _export(model, inputs, seed=0):
    torch.manual_seed(seed)
    calibration = [tuple(torch.randn_like(x) for x in inputs) for _ in range(16)]
    quantized = quantize(model.eval(), inputs, kernel_provider="ns", calibration_samples=calibration)
    return quantized, export(quantized, inputs, kernel_provider="ns", target_config=TARGET)


@pytest.mark.parametrize("affine", [True, False])
def test_quantized_layer_norm_is_lowered(affine):
    torch.manual_seed(0)
    model, x = Block(affine=affine), (torch.randn(1, 10, 16),)
    quantized, result = _export(model, x)
    assert result.edge_ops.get(QLN) == 1 and ATEN_LN not in result.edge_ops, result.edge_ops
    ops = build_manifest(result, result.executorch_program.buffer, "ns")["operators"]
    assert ops["cortex_m_ns"] == [QLN + ".out"] and ops["portable"] == []
    # Same numerics as the stock flow, which runs the layer norm in float between dequantize and quantize: the
    # integer kernel is exact for int8 outputs.
    stock = export(quantized, x, kernel_provider="arm", target_config=TARGET)
    torch.testing.assert_close(result.edge_module(*x), stock.edge_module(*x), rtol=0, atol=0)


def test_float_layer_norm_uses_the_ns_float_kernel():
    # export() quantizes a float model's boundaries, so a float layer norm reaches the NS passes only from a
    # program quantized elsewhere with the layer norm left out. Run the lowering on a float edge program: it
    # becomes cortex_m_ns::layer_norm instead of the portable aten.native_layer_norm.
    from executorch.exir import to_edge

    from nsx_cortex_m.export import _stock_edge_compile_config
    from nsx_cortex_m.pass_manager_ns import NsCortexMPassManager
    from nsx_cortex_m.passes_ns import NsLayerNormPass

    torch.manual_seed(1)
    model, x = torch.nn.LayerNorm(24), (torch.randn(2, 5, 24),)
    with torch.no_grad():
        model.weight.normal_(1.0, 0.3)
        model.bias.normal_(0.0, 0.2)
    edge = to_edge(torch.export.export(model.eval(), x, strict=True), compile_config=_stock_edge_compile_config())
    ep = NsCortexMPassManager(edge.exported_program(), passes=[NsLayerNormPass], target_config=TARGET).transform()
    targets = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
    assert any("cortex_m_ns.layer_norm" in t for t in targets), targets
    assert not any("native_layer_norm" in t for t in targets), targets
    torch.testing.assert_close(ep.module()(*x), model(*x))


def test_layer_norm_over_several_dims_stays_in_float():
    class TwoDimNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(8, 8)
            self.ln = torch.nn.LayerNorm([4, 8])
            self.out = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.out(self.ln(self.inp(x)))

    torch.manual_seed(2)
    _, result = _export(TwoDimNorm(), (torch.randn(1, 4, 8),))
    assert QLN not in result.edge_ops and FLN not in result.edge_ops
    assert result.edge_ops.get(ATEN_LN) == 1


def test_export_params_reject_what_the_kernel_cannot_represent():
    with pytest.raises(L.LayerNormNotSupported, match="row size"):
        L.export_params(None, None, 1e-5, 0.02, 0.03, 0, 40000, torch.int16)
    with pytest.raises(L.LayerNormNotSupported, match="too large|needs"):
        L.export_params(torch.full((8,), 1e6), None, 1e-5, 0.02, 1e-3, 0, 8, torch.int8)
    with pytest.raises(L.LayerNormNotSupported, match="activations"):
        L.export_params(None, None, 1e-5, 0.02, 0.03, 0, 8, torch.int32)


# -- against ns-cmsis-nn ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ns_root() -> Path:
    root = os.environ.get("NSX_EXECUTORCH_TEST_NS_CMSIS_NN_ROOT")
    if not root or not (Path(root) / "Include" / "Internal" / "arm_nn_layer_norm.h").is_file():
        pytest.skip("set NSX_EXECUTORCH_TEST_NS_CMSIS_NN_ROOT to an ns-cmsis-nn with the layer-norm kernels")
    return Path(root)


def _cases():
    rng = np.random.default_rng(20261105)
    for dtype in (torch.int8, torch.int16):
        info = torch.iinfo(dtype)
        for n in (1, 3, 16, 17, 100, 257, 768):
            for kind in ("spread", "constant", "extreme", "tiny"):
                if kind == "spread":
                    x = np.round(rng.normal(rng.normal(0, 10), info.max / 5, (3, n)))
                elif kind == "constant":
                    x = np.full((3, n), int(rng.integers(info.min, info.max)))
                elif kind == "extreme":
                    x = rng.choice([info.min, info.max], (3, n))
                else:
                    x = rng.integers(-2, 3, (3, n))
                x = torch.from_numpy(np.clip(x, info.min, info.max)).to(dtype)
                gamma = torch.from_numpy(rng.normal(1.0, 0.5, n))
                gamma[::4] *= -1
                beta = torch.from_numpy(rng.normal(0.0, 0.5, n))
                s_in = float(rng.uniform(0.005, 0.05)) / (1 if dtype == torch.int8 else 128)
                s_out = float(rng.uniform(3.0, 6.0)) / info.max
                zp = int(rng.integers(-20, 20)) if dtype == torch.int8 else 0
                eps = {"spread": 1e-5, "constant": 0.0, "extreme": 1.0, "tiny": 1e-12}[kind]
                yield dtype, x, gamma, beta, eps, s_in, s_out, zp


def test_matches_the_ns_cmsis_nn_reference(ns_root):
    spec = importlib.util.spec_from_file_location("layer_norm_reference",
                                                  ns_root / "Tests" / "UnitTest" / "layer_norm_reference.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules["layer_norm_reference"] = ref  # dataclasses look their module up there
    spec.loader.exec_module(ref)
    for dtype, x, gamma, beta, eps, s_in, s_out, zp in _cases():
        n = x.shape[-1]
        name = "s8" if dtype == torch.int8 else "s16"
        rp = ref.export_params(gamma.numpy(), beta.numpy(), eps, s_in, s_out, zp, n, name)
        p = L.export_params(gamma, beta, eps, s_in, s_out, zp, n, dtype)
        assert (p.epsilon, p.output_shift) == (rp.epsilon, rp.output_shift)
        assert np.array_equal(p.gamma.numpy(), rp.gamma) and np.array_equal(p.bias.numpy(), rp.bias)
        y = L.layer_norm_int(x, p.gamma, p.bias, p.epsilon, p.output_shift, p.activation_min, p.activation_max)
        assert np.array_equal(y.numpy(), ref.layer_norm_int(x.numpy(), rp)), (name, n)


@pytest.fixture(scope="module")
def kernel_lib(ns_root, tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        pytest.skip("no host C compiler")
    out = tmp_path_factory.mktemp("layer-norm") / "liblayernorm.so"
    src = ns_root / "Source" / "NNSupportFunctions"
    subprocess.run([cc, "-O2", "-std=c11", "-shared", "-fPIC", f"-I{ns_root / 'Include'}",
                    str(src / "arm_layer_norm_s8.c"), str(src / "arm_layer_norm_s16.c"),
                    str(src / "arm_nntables.c"), "-o", str(out)], check=True)
    return ctypes.CDLL(str(out))


class _Activation(ctypes.Structure):
    _fields_ = [("min", ctypes.c_int32), ("max", ctypes.c_int32)]


class _Params(ctypes.Structure):
    _fields_ = [("epsilon", ctypes.c_int64), ("output_shift", ctypes.c_int32), ("activation", _Activation)]


def test_matches_the_c_kernel(kernel_lib):
    """The host build of arm_layer_norm_s8 / _s16 (scalar path) against the reference the export relies on."""
    for dtype, x, gamma, beta, eps, s_in, s_out, zp in _cases():
        n = x.shape[-1]
        p = L.export_params(gamma, beta, eps, s_in, s_out, zp, n, dtype)
        expected = L.layer_norm_int(x, p.gamma, p.bias, p.epsilon, p.output_shift, p.activation_min,
                                    p.activation_max)
        params = _Params(p.epsilon, p.output_shift, _Activation(p.activation_min, p.activation_max))
        xs = x.contiguous().numpy()
        out = np.empty_like(xs)
        fn = kernel_lib.arm_layer_norm_s8 if dtype == torch.int8 else kernel_lib.arm_layer_norm_s16
        status = fn(ctypes.byref(params), ctypes.c_int32(xs.shape[0]), ctypes.c_int32(n),
                    xs.ctypes.data_as(ctypes.c_void_p),
                    p.gamma.numpy().ctypes.data_as(ctypes.c_void_p),
                    p.bias.numpy().ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p))
        assert status == 0
        assert np.array_equal(out, expected.numpy()), (dtype, n)
