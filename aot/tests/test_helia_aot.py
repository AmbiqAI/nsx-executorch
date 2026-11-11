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

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

# The delegate needs heliaAOT's PT2E front end (helia-aot >= 1.5); an older heliaAOT skips these tests too.
pytest.importorskip("helia_aot.converters.pt2e")

from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig  # noqa: E402

from nsx_cortex_m.export import export, quantize  # noqa: E402
from nsx_cortex_m.helia_aot import BACKEND_ID, DelegateBlob, HeliaAotOptions, TensorContract  # noqa: E402
from helia_aot.converters.pt2e.graph import dim_order  # noqa: E402
from nsx_cortex_m.manifest import build_manifest  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HOST_MAIN = Path(__file__).resolve().parent / "helia_aot_host" / "host_main.c"
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


class LayerScale(torch.nn.Module):
    """A learned per-channel scale, as in ConvNeXt's layer scale."""

    def __init__(self, channels):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.rand(1, channels, 1, 1) + 0.5)


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
# The cases' weights and example inputs are drawn here, at import: seeded, so a failure can be reproduced.
torch.manual_seed(0)
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
    # padding LiteRT has no name for (neither SAME nor VALID): stated by its pads alone
    "conv5x5_pad1": (Fn(lambda m, x: m["c"](x), c=torch.nn.Conv2d(4, 8, 5, padding=1)), (_cl(1, 4, 8, 8),)),
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
    # broadcasting: each operand must be laid out like the (channels_last) result
    "layer_scale": (
        Fn(lambda m, x: m["c"](x) * m["s"].scale, c=torch.nn.Conv2d(8, 8, 1), s=LayerScale(8)),
        (_cl(1, 8, 4, 5),),
    ),
    "add_channel_bias": (Fn(lambda m, a, b: a + b), (_cl(1, 8, 4, 5), torch.randn(8, 1, 1))),
    "add_width_broadcast": (Fn(lambda m, a, b: a + b), (_cl(1, 8, 4, 5), torch.randn(5))),
    "add_lower_rank_left": (Fn(lambda m, a, b: a + b), (torch.randn(8), torch.randn(3, 8))),
    # activations
    "relu": (Fn(lambda m, x: torch.relu(x)), (_cl(1, 8, 4, 4),)),
    "relu6": (Fn(lambda m, x: F.relu6(x)), (_cl(1, 8, 4, 4),)),
    "leaky_relu": (Fn(lambda m, x: F.leaky_relu(x, 0.1)), (_cl(1, 8, 4, 4),)),
    "hardswish": (Fn(lambda m, x: F.hardswish(x)), (_cl(1, 8, 4, 4),)),
    "softmax": (Fn(lambda m, x: torch.softmax(x, -1)), (torch.randn(1, 12),)),
    # pooling, reductions, data movement
    "max_pool": (Fn(lambda m, x: F.max_pool2d(x, 2)), (_cl(1, 8, 8, 8),)),
    "avg_pool": (Fn(lambda m, x: F.avg_pool2d(x, 2)), (_cl(1, 8, 8, 8),)),
    "avg_pool_padded": (Fn(lambda m, x: F.avg_pool2d(x, 3, 1, 1)), (_cl(1, 8, 8, 8),)),
    "max_pool_padded": (Fn(lambda m, x: F.max_pool2d(x, 2, 2, padding=1)), (_cl(1, 8, 6, 6),)),
    "mean": (Fn(lambda m, x: x.mean(dim=(2, 3), keepdim=True)), (torch.randn(1, 8, 4, 4),)),
    # global average pooling of a channels_last tensor: PyTorch's result is contiguous
    "mean_channels_last": (Fn(lambda m, x: x.mean(dim=(2, 3), keepdim=True)), (_cl(1, 8, 4, 4),)),
    "pad": (Fn(lambda m, x: F.pad(x, (1, 1, 2, 0))), (_cl(1, 8, 4, 4),)),
    "transpose": (Fn(lambda m, x: x.permute(0, 2, 1)), (torch.randn(1, 6, 10),)),
    "flatten_linear": (Fn(lambda m, x: m["l"](x.flatten(1)), l=torch.nn.Linear(8, 4)), (_cl(1, 8, 1, 1),)),
    # a whole model
    "kws_ds_cnn": (DsCnn(), (_cl(1, 1, 49, 10),)),
}


def _quantized(model, inputs, seed=0):
    torch.manual_seed(seed)
    calibration = [tuple(torch.randn_like(x).to(memory_format=torch.channels_last if x.dim() == 4 and
                                                  x.is_contiguous(memory_format=torch.channels_last) else
                                                  torch.contiguous_format) for x in inputs) for _ in range(8)]
    return quantize(model.eval(), inputs, kernel_provider="ns", calibration_samples=calibration)


def _delegated(quantized, inputs, out_dir: Path, capabilities=("MCU",), platform="apollo4p_evb"):
    opts = HeliaAotOptions(output_dir=out_dir, platform=platform, capabilities=list(capabilities))
    return export(quantized, inputs, kernel_provider="ns", target_config=TARGET, helia_aot=opts)


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


def test_a_module_name_the_runtime_cannot_read_fails_the_export():
    # src/helia_aot_backend.cpp reads names of up to 63 bytes; a longer one must not reach a device.
    assert DelegateBlob("m" * 63, (), ()).pack()
    with pytest.raises(ValueError, match="shorter module_prefix"):
        DelegateBlob("m" * 64, (), ()).pack()


def test_a_partition_with_a_float_boundary_is_not_a_delegate(tmp_path):
    # Lowered directly, without the partitioner, the program takes and returns float tensors. heliaAOT would
    # import it (its int8 boundary is inside), but the runtime copies the ExecuTorch tensors as they are.
    from executorch.exir import to_edge
    from helia_aot.air import AirModelSource

    from nsx_cortex_m.helia_aot.backend import HeliaAotBackend
    from nsx_cortex_m.helia_aot.codegen import compile_specs_for

    model = Fn(lambda m, x: m["c"](x), c=torch.nn.Conv2d(4, 8, 1))
    inputs = (_cl(1, 4, 6, 6),)
    edge = to_edge(_quantized(model, inputs)).exported_program()
    specs = compile_specs_for(HeliaAotOptions(output_dir=tmp_path), AirModelSource("model.pt2", "ab" * 32))
    with pytest.raises(ValueError, match="is torch.float32, not int8"):
        HeliaAotBackend.preprocess(edge, specs)


def test_the_registry_counts_the_modules_it_turned_away(tmp_path):
    # Modules register from constructors, where nothing can report a failure: the registry keeps the count for
    # HeliaAotBackend to report when a PTE names a module it cannot find.
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no host C compiler")
    main = tmp_path / "main.c"
    main.write_text(
        '#include "nsx_helia_aot.h"\n'
        "int main(void) {\n"
        '    static const nsx_helia_aot_model_t a = {.name = "a"}, b = {.name = "b"}, c = {.name = "c"};\n'
        "    if (nsx_helia_aot_register(&a) != 0 || nsx_helia_aot_register(&b) != 0) return 1;\n"
        "    if (nsx_helia_aot_register(&a) != 0 || nsx_helia_aot_dropped() != 0) return 2;\n"
        "    if (nsx_helia_aot_register(&c) != -1 || nsx_helia_aot_dropped() != 1) return 3;\n"
        '    if (nsx_helia_aot_find("c") != 0 || nsx_helia_aot_capacity() != 2) return 4;\n'
        "    return 0;\n"
        "}\n"
    )
    exe = tmp_path / "registry"
    subprocess.run([
        cc, "-std=c11", "-DNSX_HELIA_AOT_MAX_MODELS=2", f"-I{REPO / 'includes-api'}", str(main),
        str(REPO / "src" / "nsx_helia_aot_registry.c"), "-o", str(exe),
    ], check=True)
    assert subprocess.run([str(exe)]).returncode == 0


def test_the_lowering_report_says_what_the_firmware_must_link(tmp_path):
    from nsx_cortex_m.manifest import lowering_report

    model = Fn(lambda m, x: m["c"](x), c=torch.nn.Conv2d(4, 8, 1))
    inputs = (_cl(1, 4, 6, 6),)
    result = _delegated(_quantized(model, inputs), inputs, tmp_path)
    manifest = build_manifest(result, result.executorch_program.buffer, "ns")
    report = lowering_report(manifest)
    (blob,) = _blobs(result)
    assert "NSX_EXECUTORCH_ENABLE_HELIA_AOT=ON" in report and blob.module in report
    assert "NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS" not in report
    manifest["helia_aot_modules"] = [{"module": f"haot_{i}"} for i in range(9)]
    assert "NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS=9" in lowering_report(manifest)


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


@pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")
def test_firmware_build_configures_from_the_sidecar(tmp_path):
    # cmake/nsx_executorch_sidecar.cmake reads the sidecar export writes: the delegate's module is linked, the
    # backend switched on and only the remaining operators registered, with no operator names copied by hand.
    model, inputs = CASES["kws_ds_cnn"]
    result = _delegated(_quantized(model, inputs), inputs, tmp_path / "helia_aot")
    pte = tmp_path / "kws.pte"
    result.write_pte(pte)
    (blob,) = _blobs(result)
    script = Path(__file__).resolve().parents[2] / "tests" / "sidecar" / "print_config.cmake"
    run = subprocess.run(["cmake", f"-DSIDECARS={pte}.json", "-P", str(script)],
                         capture_output=True, text=True, check=True)
    config = dict(line.split("=", 1) for line in run.stderr.splitlines() if line.startswith("NSX_EXECUTORCH_"))
    assert config == {
        "NSX_EXECUTORCH_CMSIS_NN_PROVIDER": "ns",  # the delegate needs it, and none was chosen
        "NSX_EXECUTORCH_HELIA_AOT_MAX_MODELS": "",  # one module fits the registry's default
        # Only the float<->int8 conversions stay in ExecuTorch.
        "NSX_EXECUTORCH_CORTEX_M_SELECT_OPS_LIST": "cortex_m::dequantize_per_tensor.out,cortex_m::quantize_per_tensor.out",
        "NSX_EXECUTORCH_NS_SELECT_OPS_LIST": "NONE",
        "NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST": "",
        "NSX_EXECUTORCH_ENABLE_NS_OPS": "",
        "NSX_EXECUTORCH_ENABLE_HELIA_AOT": "ON",
        "NSX_EXECUTORCH_HELIA_AOT_MODULES": blob.module,
    }

    # A re-exported PTE that the sidecar no longer describes fails the configure.
    pte.write_bytes(pte.read_bytes() + b"\0")
    stale = subprocess.run(["cmake", f"-DSIDECARS={pte}.json", "-P", str(script)], capture_output=True, text=True)
    assert stale.returncode != 0 and "Re-export the model" in " ".join(stale.stderr.split())


def test_unsupported_ops_stay_in_executorch(tmp_path):
    # hardtanh(-2, 3) is not an AIR ReLU variant: the convolutions around it are delegated, it is not.
    model = Fn(lambda m, x: m["b"](F.hardtanh(m["a"](x), -2.0, 3.0)),
               a=torch.nn.Conv2d(4, 8, 3, padding=1), b=torch.nn.Conv2d(8, 8, 1))
    inputs = (_cl(1, 4, 6, 6),)
    result = _delegated(_quantized(model, inputs), inputs, tmp_path)
    assert len(_blobs(result)) >= 1
    ops = build_manifest(result, result.executorch_program.buffer, "ns")["operators"]
    assert not any("conv" in op for op in ops["cortex_m"] + ops["cortex_m_ns"])


def test_each_partition_is_named_by_the_saved_program(tmp_path):
    # The quantized program is saved next to the modules; each partition's module name and heliaAOT plan name it by
    # that file's SHA-256 and the partition's index, so a plan made for a partition matches the same file again.
    import hashlib
    import json

    # The sigmoid has no int8 form, so the convolutions around it are two partitions.
    model = Fn(lambda m, x: m["b"](torch.sigmoid(m["a"](x))),
               a=torch.nn.Conv2d(4, 8, 3, padding=1), b=torch.nn.Conv2d(8, 8, 1))
    inputs = (_cl(1, 4, 6, 6),)
    result = _delegated(_quantized(model, inputs), inputs, tmp_path)
    (saved,) = tmp_path.glob("haot_*.pt2")
    digest = hashlib.sha256(saved.read_bytes()).hexdigest()
    assert saved.name == f"haot_{digest[:10]}.pt2"
    assert torch.export.load(saved) is not None  # the program itself: helia-torch compile can rebuild from it
    modules = sorted(blob.module for blob in _blobs(result))
    assert modules == [f"haot_{digest[:10]}_0", f"haot_{digest[:10]}_1"]
    for index, module in enumerate(modules):
        plan = json.loads((tmp_path / module / f"{module}_plan.json").read_text())
        assert plan["model"] == {"file": saved.name, "sha256": digest, "subgraph": index}


def _seeded_conv(seed):
    torch.manual_seed(seed)
    return Fn(lambda m, x: torch.relu(m["c"](x)), c=torch.nn.Conv2d(4, 8, 3, padding=1)), (_cl(1, 4, 6, 6),)


def _export_into(directory, seed, **options):
    model, inputs = _seeded_conv(seed)
    opts = HeliaAotOptions(output_dir=directory, platform="apollo4p_evb", capabilities=["MCU"], **options)
    result = export(_quantized(model, inputs), inputs, kernel_provider="ns", target_config=TARGET, helia_aot=opts)
    (saved,) = [p for p in directory.glob("haot_*.pt2") if not p.name.endswith(".debug.pt2")]
    return saved, sorted(blob.module for blob in _blobs(result))


def test_exporting_the_same_model_again_gives_the_same_program_file_and_modules(tmp_path):
    # The saved program leaves out the debug metadata PyTorch writes differently on every export, so its bytes,
    # and with them the digest in the module names and plans, depend only on the quantized program.
    first, first_modules = _export_into(tmp_path / "first", seed=0)
    again, again_modules = _export_into(tmp_path / "again", seed=0)
    assert first.name == again.name and first.read_bytes() == again.read_bytes()
    assert first_modules == again_modules
    other, other_modules = _export_into(tmp_path / "other", seed=1)  # other weights: another model
    assert other.name != first.name and other_modules != first_modules


def test_the_debug_program_keeps_the_metadata_the_saved_program_leaves_out(tmp_path):
    saved, _ = _export_into(tmp_path, seed=0, debug_program=True)
    debug = saved.with_suffix(".debug.pt2")
    assert debug.is_file()
    traced = ("from_node", "nn_module_stack", "stack_trace")

    def carrying(path):
        return sum(1 for node in torch.export.load(path).graph.nodes if any(node.meta.get(key) for key in traced))

    assert carrying(saved) == 0 and carrying(debug) > 0
    # Without the option only the reproducible file is written.
    plain, _ = _export_into(tmp_path / "plain", seed=0)
    assert not plain.with_suffix(".debug.pt2").exists() and plain.read_bytes() == saved.read_bytes()


def _dequantize_count(result):
    graph = result.executorch_program.exported_program().graph
    return sum(1 for n in graph.nodes if n.op == "call_function" and "dequantize_per_tensor" in str(n.target))


def test_a_dequantize_shared_outside_the_delegate_exports_and_stays_shared(tmp_path):
    # sigmoid and tanh read one dequantized value and both stay in ExecuTorch. Each consumer gets its own
    # dequantize before partitioning (ExecuTorch refuses a partitioner that changes the graph), and the copies no
    # partition took are merged again, so the program runs as many dequantize kernels as without the delegate.
    model = Fn(lambda m, x: (lambda a: torch.sigmoid(a) + torch.tanh(a))(m["a"](x)), a=torch.nn.Conv2d(4, 8, 1))
    inputs = (_cl(1, 4, 6, 6),)
    quantized = _quantized(model, inputs)
    delegated = _delegated(quantized, inputs, tmp_path)
    plain = export(quantized, inputs, kernel_provider="ns", target_config=TARGET)
    assert len(_blobs(delegated)) == 2  # the convolution, and the add
    assert _dequantize_count(delegated) == _dequantize_count(plain)


def test_the_pte_holds_the_blob_and_none_of_the_build_options(tmp_path):
    # ExecuTorch writes a delegate's compile specs into the PTE; they named the output directory, the platform and
    # the saved program. The runtime reads only the blob, so they are dropped before the program is emitted.
    out_dir = tmp_path / "build_machine_only_path"
    model, inputs = CASES["relu"]
    result = _delegated(_quantized(model, inputs), inputs, out_dir)
    pte = bytes(result.executorch_program.buffer)
    (blob,) = _blobs(result)
    assert blob.module.encode() in pte
    for leaked in (str(out_dir), "build_machine_only_path", "output_dir", "source_sha256", "apollo4p_evb"):
        assert leaked.encode() not in pte, leaked


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


def _build_host_runner(cc, ns_root, ns_lib, module: Path, name: str, out: Path) -> Path:
    platform_h = (module / "includes-api" / f"{name}_platform.h").read_text()
    placement = sorted(set(re.findall(r"#ifndef\s+(\w+_PUT_\w+)", platform_h)))
    sources = [str(p) for p in sorted((module / "src").glob("*.c")) if not p.name.endswith("_test_case.c")]
    exe = out / f"{name}_host"
    subprocess.run([
        cc, "-O1", "-std=c11", "-w", "-DARM_NN_ENABLE_F32=1", *(f"-D{p}=" for p in placement),
        f"-I{module / 'includes-api'}", f"-I{REPO / 'includes-api'}", f"-I{ns_root / 'Include'}",
        f"-I{ns_root / 'Include' / 'Internal'}", str(HOST_MAIN), str(REPO / "src" / "nsx_helia_aot_registry.c"),
        str(module / "et" / f"{name}_et_registration.c"), *sources, str(ns_lib), "-lm", "-o", str(exe),
    ], check=True)
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
# Cases ExecuTorch leaves on the float kernels, which the delegate computes in int8: the dims of the mean.
# nsx-executorch lowers a mean to cortex_m_ns::quantized_mean (arm_mean_s8) only for a contiguous tensor, so
# without the delegate a channels_last one runs aten.mean in float, which the int8 kernel matches to 1 LSB. The
# delegate is held to the int8 operator's reference, as the contiguous "mean" case is.
MEAN_ORACLES = {"mean_channels_last": (2, 3)}


def _mean_kernel_oracle(x_q, dims, in_scale, in_zp, out_scale, out_zp) -> torch.Tensor:
    from nsx_cortex_m.operators_ns import quantized_mean_impl
    from nsx_cortex_m.passes_ns import quantize_multiplier_aot

    count = 1
    for d in dims:
        count *= x_q.shape[d]
    multiplier, shift = quantize_multiplier_aot(in_scale / (out_scale * count))
    mask = [1 if d in dims else 0 for d in range(4)]
    return quantized_mean_impl(x_q, in_zp, mask, True, out_zp, int(multiplier), int(shift))


def _physical_bytes(t: torch.Tensor, order) -> bytes:
    return t.permute(tuple(order)).contiguous().numpy().tobytes()


@pytest.mark.parametrize("name", list(CASES))
def test_bit_exact_on_host(name, tmp_path, ns_cmsis_nn_lib):
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
    exe = _build_host_runner(cc, ns_root, ns_lib, tmp_path / "modules" / blob.module, blob.module, tmp_path)

    gen = torch.Generator().manual_seed(1234)
    for trial in range(4):
        xs = [(torch.randn(x.shape, generator=gen) * (1 + trial)).to(
            memory_format=torch.channels_last if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format) for x in inputs]
        args = [str(blob.module), str(len(xs))]
        for i, (x, q) in enumerate(zip(xs, q_nodes)):
            _, scale, zp, qmin, qmax, dtype = q.args
            x_q = torch.ops.cortex_m.quantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
            path = tmp_path / f"in{i}.bin"
            path.write_bytes(_physical_bytes(x_q, dim_order(q.meta["val"])))
            args.append(str(path))
        ys = reference.edge_module(*xs)
        ys = ys if isinstance(ys, (tuple, list)) else (ys,)
        args.append(str(len(dq_nodes)))
        expected = []
        for o, (y, dq) in enumerate(zip(ys, dq_nodes)):
            _, scale, zp, qmin, qmax, _ = dq.args
            if name in KERNEL_ORACLES:
                kernel, stride, pad = KERNEL_ORACLES[name]
                in_zp = q_nodes[0].args[2]
                assert (q_nodes[0].args[1], in_zp) == (scale, zp), "pooling shares its input quantization"
                y_q = _avg_pool_kernel_oracle(x_q, kernel, stride, pad, in_zp)
            elif name in MEAN_ORACLES:
                in_scale, in_zp = q_nodes[0].args[1:3]
                y_q = _mean_kernel_oracle(x_q, MEAN_ORACLES[name], in_scale, in_zp, scale, zp)
                float_q = torch.clamp(torch.round(y / scale) + zp, qmin, qmax)
                assert (y_q.to(torch.int64) - float_q.to(torch.int64)).abs().max() <= 1
            else:
                y_q = torch.clamp(torch.round(y / scale) + zp, qmin, qmax).to(torch.int8)
                assert torch.equal((y_q.float() - zp) * scale, y), "reference output is not on its int8 grid"
            expected.append(_physical_bytes(y_q, dim_order(dq.args[0].meta["val"])))
            args.append(f"{tmp_path / f'out{o}.bin'}:{len(expected[-1])}")
        subprocess.run([str(exe), *args[0:1], *args[1:]], check=True)
        for o, exp in enumerate(expected):
            got = (tmp_path / f"out{o}.bin").read_bytes()
            diffs = sum(a != b for a, b in zip(got, exp))
            assert got == exp, f"{name} trial {trial} output {o}: {diffs}/{len(exp)} bytes differ"
