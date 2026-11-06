# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# helia-aot delegate: partitioning, the delegate blob, and per-operator
# bit-exactness of heliaAOT-generated code against ExecuTorch's reference
# kernels.
#
# The bit-exact tests export each model twice from one quantized program --
# delegated to heliaAOT (generated for a plain-C target, capabilities MCU) and
# not delegated -- then compile the generated module with ns-cmsis-nn on the
# host and compare its int8 outputs with the reference kernels' byte for byte.
# CMSIS-NN's portable and MVE kernels are bit-exact by design; the on-device
# check of the MVE build is firmware-side.
#
# Needs: helia-aot, a host C compiler, and NSX_EXECUTORCH_TEST_NS_CMSIS_NN_ROOT
# pointing at an ns-cmsis-nn >= 7.47.0 checkout (the bit-exact tests skip
# otherwise).

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

pytest.importorskip("helia_aot")

from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig  # noqa: E402

from nsx_cortex_m.export import export, quantize  # noqa: E402
from nsx_cortex_m.helia_aot import (  # noqa: E402
    BACKEND_ID,
    DelegateBlob,
    HeliaAotOptions,
    NotFullyDelegableError,
    TensorContract,
    compile_model,
)
from nsx_cortex_m.helia_aot.frontend import dim_order  # noqa: E402
from nsx_cortex_m.manifest import build_manifest  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HOST_MAIN = Path(__file__).resolve().parent / "helia_aot_host" / "host_main.c"
STANDALONE_MAIN = Path(__file__).resolve().parent / "helia_aot_host" / "standalone_main.c"
TARGET = CortexMTargetConfig(cpu=CortexM.M55)


def _cl(*shape):
    return torch.randn(*shape).to(memory_format=torch.channels_last)


class Fn(torch.nn.Module):
    def __init__(self, fn, **modules):
        super().__init__()
        self.fn = fn
        self.mods = torch.nn.ModuleDict(modules)

    def forward(self, *x):
        return self.fn(self.mods, *x)


class DsCnn(torch.nn.Module):
    """MLPerf Tiny KWS DS-CNN."""

    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(1, 64, (10, 4), stride=2, padding=(5, 1))
        self.dw = torch.nn.ModuleList(torch.nn.Conv2d(64, 64, 3, padding=1, groups=64) for _ in range(4))
        self.pw = torch.nn.ModuleList(torch.nn.Conv2d(64, 64, 1) for _ in range(4))
        self.pool = torch.nn.AvgPool2d((25, 5))
        self.fc = torch.nn.Linear(64, 12)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        for dw, pw in zip(self.dw, self.pw):
            x = torch.relu(pw(torch.relu(dw(x))))
        return self.fc(self.pool(x).flatten(1))


F = torch.nn.functional


def _layer_norm(n):
    ln = torch.nn.LayerNorm(n)
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.3)
        ln.weight[::5] *= -1
        ln.bias.normal_(0.0, 0.2)
    return ln


CASES = {
    # convolutions and matmuls
    "conv3x3_relu": (Fn(lambda m, x: torch.relu(m["c"](x)), c=torch.nn.Conv2d(8, 16, 3, padding=1)), (_cl(1, 8, 9, 7),)),
    "conv_stride2": (Fn(lambda m, x: m["c"](x), c=torch.nn.Conv2d(4, 8, 3, stride=2, padding=1)), (_cl(1, 4, 10, 10),)),
    "conv1x1": (Fn(lambda m, x: m["c"](x), c=torch.nn.Conv2d(16, 32, 1)), (_cl(1, 16, 5, 5),)),
    "conv_small_cin": (Fn(lambda m, x: torch.relu(m["c"](x)), c=torch.nn.Conv2d(1, 16, (10, 4), stride=2, padding=(5, 1))),
                       (_cl(1, 1, 49, 10),)),
    "depthwise3x3": (Fn(lambda m, x: torch.relu(m["c"](x)), c=torch.nn.Conv2d(32, 32, 3, padding=1, groups=32)),
                     (_cl(1, 32, 12, 12),)),
    "depthwise_mult2": (Fn(lambda m, x: m["c"](x), c=torch.nn.Conv2d(8, 16, 3, padding=1, groups=8)), (_cl(1, 8, 6, 6),)),
    "transpose_conv": (Fn(lambda m, x: m["c"](x), c=torch.nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1)),
                       (_cl(1, 4, 5, 5),)),
    "linear": (Fn(lambda m, x: m["l"](x), l=torch.nn.Linear(32, 10)), (torch.randn(1, 32),)),
    "bmm": (Fn(lambda m, a, b: torch.bmm(a, b)), (torch.randn(1, 6, 8), torch.randn(1, 8, 5))),
    # elementwise
    "add": (Fn(lambda m, a, b: a + b), (_cl(1, 8, 4, 4), _cl(1, 8, 4, 4))),
    "sub": (Fn(lambda m, a, b: a - b), (_cl(1, 8, 4, 4), _cl(1, 8, 4, 4))),
    "mul": (Fn(lambda m, a, b: a * b), (_cl(1, 8, 4, 4), _cl(1, 8, 4, 4))),
    "maximum": (Fn(lambda m, a, b: torch.maximum(a, b)), (_cl(1, 8, 4, 4), _cl(1, 8, 4, 4))),
    "minimum": (Fn(lambda m, a, b: torch.minimum(a, b)), (_cl(1, 8, 4, 4), _cl(1, 8, 4, 4))),
    # activations
    "relu": (Fn(lambda m, x: torch.relu(x)), (_cl(1, 8, 4, 4),)),
    "leaky_relu": (Fn(lambda m, x: F.leaky_relu(x, 0.1)), (_cl(1, 8, 4, 4),)),
    "hardswish": (Fn(lambda m, x: F.hardswish(x)), (_cl(1, 8, 4, 4),)),
    "softmax": (Fn(lambda m, x: torch.softmax(x, -1)), (torch.randn(1, 12),)),
    # pooling, reductions, data movement
    "max_pool": (Fn(lambda m, x: F.max_pool2d(x, 2)), (_cl(1, 8, 8, 8),)),
    "avg_pool": (Fn(lambda m, x: F.avg_pool2d(x, 2)), (_cl(1, 8, 8, 8),)),
    "avg_pool_padded": (Fn(lambda m, x: F.avg_pool2d(x, 3, 1, 1)), (_cl(1, 8, 8, 8),)),
    "mean": (Fn(lambda m, x: x.mean(dim=(2, 3), keepdim=True)), (torch.randn(1, 8, 4, 4),)),
    "pad": (Fn(lambda m, x: F.pad(x, (1, 1, 2, 0))), (_cl(1, 8, 4, 4),)),
    "transpose": (Fn(lambda m, x: x.permute(0, 2, 1)), (torch.randn(1, 6, 10),)),
    "flatten_linear": (Fn(lambda m, x: m["l"](x.flatten(1)), l=torch.nn.Linear(8, 4)), (_cl(1, 8, 1, 1),)),
    # normalization (needs a heliaAOT with LAYER_NORM)
    "layer_norm": (Fn(lambda m, x: m["ln"](x), ln=_layer_norm(24)), (torch.randn(1, 6, 24),)),
    "transformer_mlp": (Fn(lambda m, x: m["down"](torch.relu(m["ln"](m["up"](x)))), up=torch.nn.Linear(16, 32),
                           ln=_layer_norm(32), down=torch.nn.Linear(32, 16)), (torch.randn(1, 10, 16),)),
    # a whole model
    "kws_ds_cnn": (DsCnn(), (_cl(1, 1, 49, 10),)),
}
NEEDS_LAYER_NORM = {"layer_norm", "transformer_mlp"}


def _skip_without_layer_norm(name):
    if name in NEEDS_LAYER_NORM:
        try:
            from helia_aot.air import AirLayerNormOptions  # noqa: F401
        except ImportError:
            pytest.skip("this heliaAOT has no LAYER_NORM operator")


def _quantized(model, inputs, seed=0):
    torch.manual_seed(seed)
    calibration = [tuple(torch.randn_like(x).to(memory_format=torch.channels_last if x.dim() == 4 and
                                                  x.is_contiguous(memory_format=torch.channels_last) else
                                                  torch.contiguous_format) for x in inputs) for _ in range(8)]
    return quantize(model.eval(), inputs, kernel_provider="ns", calibration_samples=calibration)


def _host_options(out_dir: Path) -> HeliaAotOptions:
    # Plain-C code generation, so the module runs on the host.
    return HeliaAotOptions(output_dir=out_dir, platform="apollo4p_evb", capabilities=["MCU"])


def _delegated(quantized, inputs, out_dir: Path):
    return export(quantized, inputs, kernel_provider="ns", target_config=TARGET, helia_aot=_host_options(out_dir))


def _boundaries(result):
    """(input quantize nodes in input order, output dequantize nodes in output order) of a delegated program."""
    ep = result.executorch_program.exported_program()
    graph = ep.graph_module.graph
    user_inputs = [s.arg.name for s in ep.graph_signature.input_specs if s.kind.name == "USER_INPUT"]
    by_name = {n.name: n for n in graph.nodes}
    q_nodes = []
    for name in user_inputs:
        users = [u for u in by_name[name].users if "quantize_per_tensor" in str(u.target)]
        assert len(users) == 1, f"input {name} is not quantized once"
        q_nodes.append(users[0])
    out = next(n for n in graph.nodes if n.op == "output")
    dq_nodes = list(out.args[0])
    return q_nodes, dq_nodes


def _blobs(result) -> list[DelegateBlob]:
    program = result.executorch_program._emitter_output.program
    plan = program.execution_plan[0]
    blobs = []
    for d in plan.delegates:
        assert d.id == BACKEND_ID
        blobs.append(DelegateBlob.unpack(bytes(program.backend_delegate_data[d.processed.index].data)))
    return blobs


# -- unit tests -----------------------------------------------------------------------------------------------


def test_blob_round_trip():
    blob = DelegateBlob("haot_0123456789", (TensorContract(1, 490),), (TensorContract(1, 12), TensorContract(6, 48)))
    packed = blob.pack()
    assert packed[:4] == b"HAOT" and len(packed) % 4 == 0
    assert DelegateBlob.unpack(packed) == blob


def test_kws_is_fully_delegated(tmp_path):
    model, inputs = CASES["kws_ds_cnn"]
    result = _delegated(_quantized(model, inputs), inputs, tmp_path)
    manifest = build_manifest(result, result.executorch_program.buffer, "ns")
    # Only the float<->int8 conversions stay in ExecuTorch.
    assert manifest["operators"]["cortex_m"] == ["cortex_m::dequantize_per_tensor.out",
                                                 "cortex_m::quantize_per_tensor.out"]
    assert manifest["operators"]["cortex_m_ns"] == [] and manifest["operators"]["portable"] == []
    assert manifest["requires_helia_aot"] is True
    (blob,) = _blobs(result)
    assert [d["module"] for d in manifest["helia_aot_modules"]] == [blob.module]
    assert blob.inputs == (TensorContract(1, 490),) and blob.outputs == (TensorContract(1, 12),)
    module = tmp_path / blob.module
    assert (module / "nsx-module.yaml").is_file()
    assert (module / "et" / f"{blob.module}_et_registration.c").is_file()
    assert "target_sources(" in (module / "CMakeLists.txt").read_text()


def _conv_sigmoid_conv():
    # A float sigmoid between two quantized convolutions: aten.sigmoid has no AIR mapping (nor a cortex_m
    # kernel), so it runs on the portable float kernel between dequantize and quantize.
    model = Fn(lambda m, x: m["b"](torch.sigmoid(m["a"](x))), a=torch.nn.Conv2d(4, 8, 1), b=torch.nn.Conv2d(8, 8, 1))
    return model, (_cl(1, 4, 6, 6),)


def test_unsupported_ops_stay_in_executorch(tmp_path):
    # The convolutions on either side are delegated; the sigmoid and its float conversions are not.
    model, inputs = _conv_sigmoid_conv()
    result = _delegated(_quantized(model, inputs), inputs, tmp_path)
    assert len(_blobs(result)) == 2
    ops = build_manifest(result, result.executorch_program.buffer, "ns")["operators"]
    assert any("sigmoid" in op for op in ops["portable"]), ops
    assert not any("conv" in op for op in ops["cortex_m"] + ops["cortex_m_ns"] + ops["portable"]), ops


def test_arm_provider_rejects_helia_aot(tmp_path):
    model, inputs = CASES["relu"]
    with pytest.raises(ValueError, match="kernel_provider='ns'"):
        export(model.eval(), inputs, kernel_provider="arm", helia_aot=HeliaAotOptions(output_dir=tmp_path))


# -- host bit-exactness ---------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ns_cmsis_nn_lib(tmp_path_factory):
    root = os.environ.get("NSX_EXECUTORCH_TEST_NS_CMSIS_NN_ROOT")
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not root or not cc:
        pytest.skip("set NSX_EXECUTORCH_TEST_NS_CMSIS_NN_ROOT (ns-cmsis-nn >= 7.47) and provide a host C compiler")
    root = Path(root)
    out = tmp_path_factory.mktemp("ns-cmsis-nn")
    sources = sorted(str(p) for p in (root / "Source").rglob("*.c"))
    flags = ["-O1", "-std=c11", "-w", "-DARM_NN_ENABLE_F32=1", f"-I{root / 'Include'}", f"-I{root / 'Include' / 'Internal'}"]
    subprocess.run([cc, "-c", *flags, *sources], cwd=out, check=True)
    lib = out / "libns_cmsis_nn.a"
    subprocess.run(["ar", "rcs", str(lib), *sorted(str(p) for p in out.glob("*.o"))], check=True)
    return cc, root, lib


def _module_build_args(ns_root, module: Path, name: str) -> list[str]:
    platform_h = (module / "includes-api" / f"{name}_platform.h").read_text()
    placement = sorted(set(re.findall(r"#ifndef\s+(\w+_PUT_\w+)", platform_h)))
    sources = [str(p) for p in sorted((module / "src").glob("*.c")) if not p.name.endswith("_test_case.c")]
    return ["-O1", "-std=c11", "-w", "-DARM_NN_ENABLE_F32=1", *(f"-D{p}=" for p in placement),
            f"-I{module / 'includes-api'}", f"-I{ns_root / 'Include'}", f"-I{ns_root / 'Include' / 'Internal'}",
            *sources]


def _build_host_runner(cc, ns_root, ns_lib, module: Path, name: str, out: Path) -> Path:
    """The delegate's view of a module: found by name through its registration and the runtime registry."""
    exe = out / f"{name}_host"
    subprocess.run([
        cc, *_module_build_args(ns_root, module, name), f"-I{REPO / 'includes-api'}", str(HOST_MAIN),
        str(REPO / "src" / "nsx_helia_aot_registry.c"), str(module / "et" / f"{name}_et_registration.c"),
        str(ns_lib), "-lm", "-o", str(exe),
    ], check=True)
    return exe


def _build_standalone_runner(cc, ns_root, ns_lib, module: Path, name: str, out: Path) -> Path:
    """An application's view of a compile_model() module: <name>_model_init/_run, nothing else."""
    exe = out / f"{name}_standalone"
    subprocess.run([cc, *_module_build_args(ns_root, module, name), f"-DHAOT_NAME={name}", str(STANDALONE_MAIN),
                    str(ns_lib), "-lm", "-o", str(exe)], check=True)
    return exe


def _avg_pool_kernel_oracle(x_q_nchw: torch.Tensor, kernel: int, stride: int, pad: int, pad_value: int) -> torch.Tensor:
    """What ExecuTorch's cortex_m::quantized_avg_pool2d computes on the device: the kernel calls
    arm_avgpool_s8, which sums int8 values (after helia-torch's explicit pad with the zero point) and rounds
    half away from zero. Its Python reference (operators.py) averages dequantized floats and rounds half to
    even instead, so the two differ on exact .5 ties (common for 2x2 windows); the delegate must match the
    kernel."""
    x = torch.nn.functional.pad(x_q_nchw.to(torch.int64), (pad, pad, pad, pad), value=pad_value)
    windows = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    sums = windows.sum(dim=(-1, -2))
    count = kernel * kernel
    half = count // 2
    rounded = torch.where(sums > 0, (sums + half) // count, -((-(sums - half)) // count))
    return rounded.clamp(-128, 127).to(torch.int8)


# Cases whose ExecuTorch Python reference does not mirror its device kernel: (kernel, stride, pad).
KERNEL_ORACLES = {"avg_pool": (2, 2, 0), "avg_pool_padded": (3, 1, 1)}


def _physical_bytes(t: torch.Tensor, order) -> bytes:
    return t.permute(tuple(order)).contiguous().numpy().tobytes()


@dataclass(frozen=True)
class _Boundary:
    """One int8 boundary tensor: (de)quantization and its physical layout."""

    scale: float
    zp: int
    qmin: int
    qmax: int
    order: tuple[int, ...]


def _run_trials(name, inputs, reference, in_bounds, out_bounds, run, tmp_path, trials=4):
    """Feed `run` quantized random inputs and compare its int8 outputs with the reference kernels'."""
    gen = torch.Generator().manual_seed(1234)
    for trial in range(trials):
        xs = [(torch.randn(x.shape, generator=gen) * (1 + trial)).to(
            memory_format=torch.channels_last if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format) for x in inputs]
        in_paths, x_qs = [], []
        for i, (x, b) in enumerate(zip(xs, in_bounds)):
            x_q = torch.ops.cortex_m.quantize_per_tensor(x, b.scale, b.zp, b.qmin, b.qmax, torch.int8)
            x_qs.append(x_q)
            path = tmp_path / f"in{i}.bin"
            path.write_bytes(_physical_bytes(x_q, b.order))
            in_paths.append(path)
        ys = reference.edge_module(*xs)
        ys = ys if isinstance(ys, (tuple, list)) else (ys,)
        expected = []
        for y, b in zip(ys, out_bounds):
            if name in KERNEL_ORACLES:
                kernel, stride, pad = KERNEL_ORACLES[name]
                assert (in_bounds[0].scale, in_bounds[0].zp) == (b.scale, b.zp), "pooling shares its input quantization"
                y_q = _avg_pool_kernel_oracle(x_qs[0], kernel, stride, pad, in_bounds[0].zp)
            else:
                y_q = torch.clamp(torch.round(y / b.scale) + b.zp, b.qmin, b.qmax).to(torch.int8)
                assert torch.equal((y_q.float() - b.zp) * b.scale, y), "reference output is not on its int8 grid"
            expected.append(_physical_bytes(y_q, b.order))
        out_paths = [tmp_path / f"out{o}.bin" for o in range(len(expected))]
        run(in_paths, [(p, len(e)) for p, e in zip(out_paths, expected)])
        for o, (path, exp) in enumerate(zip(out_paths, expected)):
            got = path.read_bytes()
            diffs = sum(a != b for a, b in zip(got, exp))
            assert got == exp, f"{name} trial {trial} output {o}: {diffs}/{len(exp)} bytes differ"


def _runner_args(in_paths, out_specs) -> list[str]:
    return [str(len(in_paths)), *map(str, in_paths), str(len(out_specs)), *(f"{p}:{n}" for p, n in out_specs)]


@pytest.mark.parametrize("name", list(CASES))
def test_bit_exact_on_host(name, tmp_path, ns_cmsis_nn_lib):
    _skip_without_layer_norm(name)
    cc, ns_root, ns_lib = ns_cmsis_nn_lib
    model, inputs = CASES[name]
    quantized = _quantized(model, inputs)
    delegated = _delegated(quantized, inputs, tmp_path / "modules")
    reference = export(quantized, inputs, kernel_provider="ns", target_config=TARGET)

    (blob,) = _blobs(delegated)  # fully delegated: one partition
    ops = build_manifest(delegated, delegated.executorch_program.buffer, "ns")["operators"]
    assert set(ops["cortex_m"]) <= {"cortex_m::quantize_per_tensor.out", "cortex_m::dequantize_per_tensor.out"}
    assert not ops["cortex_m_ns"] and not ops["portable"], ops
    q_nodes, dq_nodes = _boundaries(delegated)
    in_bounds = [_Boundary(*map(float, q.args[1:2]), *map(int, q.args[2:5]), dim_order(q.meta["val"]))
                 for q in q_nodes]
    out_bounds = [_Boundary(*map(float, dq.args[1:2]), *map(int, dq.args[2:5]), dim_order(dq.args[0].meta["val"]))
                  for dq in dq_nodes]
    exe = _build_host_runner(cc, ns_root, ns_lib, tmp_path / "modules" / blob.module, blob.module, tmp_path)

    def run(in_paths, out_specs):
        subprocess.run([str(exe), blob.module, *_runner_args(in_paths, out_specs)], check=True)

    _run_trials(name, inputs, reference, in_bounds, out_bounds, run, tmp_path)


# -- compile_model: the whole model in heliaAOT, no ExecuTorch runtime ------------------------------------------


def test_compile_model_writes_a_plain_module(tmp_path):
    model, inputs = CASES["kws_ds_cnn"]
    # The quantized program as a user would hand it over: saved and loaded as a .pt2.
    torch.export.save(_quantized(model, inputs), tmp_path / "kws.pt2")
    compiled = compile_model(torch.export.load(tmp_path / "kws.pt2"), None, _host_options(tmp_path / "out"),
                             name="kws")
    module = tmp_path / "out" / "kws"
    assert compiled.module_dir == module and (module / "nsx-module.yaml").is_file()
    assert (module / "includes-api" / "kws_model.h").is_file()
    # No ExecuTorch registration: a plain heliaAOT module.
    assert not (module / "et").exists() and "nsx_helia_aot" not in (module / "CMakeLists.txt").read_text()
    (x,) = compiled.inputs
    (y,) = compiled.outputs
    assert (x.shape, x.dim_order, x.physical_shape, x.dtype) == ((1, 1, 49, 10), (0, 2, 3, 1), (1, 49, 10, 1), "int8")
    assert (y.shape, y.physical_shape, y.nbytes) == ((1, 12), (1, 12), 12)
    manifest = json.loads((module / "kws_io.json").read_text())
    assert manifest["module"] == "kws" and manifest["inputs"][0]["scale"] == x.scale


def test_io_tensor_quantize_matches_the_boundary_kernel(tmp_path):
    model, inputs = CASES["conv3x3_relu"]
    compiled = compile_model(_quantized(model, inputs), None, _host_options(tmp_path))
    (x_spec,) = compiled.inputs
    x = inputs[0] * 3
    kernel = torch.ops.cortex_m.quantize_per_tensor(x, x_spec.scale, x_spec.zero_point, x_spec.qmin, x_spec.qmax,
                                                    torch.int8)
    assert torch.equal(x_spec.quantize(x), kernel.permute(x_spec.dim_order).contiguous())
    (y_spec,) = compiled.outputs
    q = torch.randint(-128, 128, y_spec.physical_shape, dtype=torch.int8)
    inverse = [y_spec.dim_order.index(d) for d in range(4)]
    assert torch.equal(y_spec.dequantize(q), (q.permute(inverse).float() - y_spec.zero_point) * y_spec.scale)


def test_compile_model_refuses_a_partially_supported_model(tmp_path):
    model, inputs = _conv_sigmoid_conv()
    with pytest.raises(NotFullyDelegableError) as info:
        compile_model(_quantized(model, inputs), None, _host_options(tmp_path))
    assert "aten.sigmoid has no AIR mapping" in info.value.rejected.get("aten_sigmoid_default", "")
    assert "export(..., helia_aot=" in str(info.value)
    assert not any(tmp_path.iterdir()), "nothing is generated for a refused model"


def test_compile_model_rejects_a_bad_name(tmp_path):
    model, inputs = CASES["relu"]
    with pytest.raises(ValueError, match="C identifier"):
        compile_model(model.eval(), inputs, _host_options(tmp_path), name="kws-v1")


@pytest.mark.parametrize("name", list(CASES))
def test_compile_model_bit_exact_on_host(name, tmp_path, ns_cmsis_nn_lib):
    """The standalone module computes what the non-delegated ExecuTorch program does, byte for byte -- and so,
    with test_bit_exact_on_host, what the delegate does."""
    _skip_without_layer_norm(name)
    cc, ns_root, ns_lib = ns_cmsis_nn_lib
    model, inputs = CASES[name]
    quantized = _quantized(model, inputs)
    reference = export(quantized, inputs, kernel_provider="ns", target_config=TARGET)
    module_name = f"m_{name}"
    compiled = compile_model(quantized, None, _host_options(tmp_path / "modules"), name=module_name)
    exe = _build_standalone_runner(cc, ns_root, ns_lib, compiled.module_dir, module_name, tmp_path)
    in_bounds = [_Boundary(t.scale, t.zero_point, t.qmin, t.qmax, t.dim_order) for t in compiled.inputs]
    out_bounds = [_Boundary(t.scale, t.zero_point, t.qmin, t.qmax, t.dim_order) for t in compiled.outputs]

    def run(in_paths, out_specs):
        subprocess.run([str(exe), *_runner_args(in_paths, out_specs)], check=True)

    _run_trials(name, inputs, reference, in_bounds, out_bounds, run, tmp_path)
