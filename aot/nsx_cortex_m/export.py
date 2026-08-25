# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Export entry point for nsx-executorch models.
#
# kernel_provider="arm" reproduces the stock Cortex-M flow exactly (stock
# quantizer, stock pass list, stock to_edge config) — graphs are unchanged
# from upstream behavior and contain no cortex_m_ns:: ops.
#
# kernel_provider="ns" additionally lowers sub, hardswish, mean, standalone
# relu/relu6/hardtanh/clamp and leaky_relu to cortex_m_ns:: kernels backed by
# ns-cmsis-nn. The resulting PTE requires a runtime built with
# NSX_EXECUTORCH_CMSIS_NN_PROVIDER=ns and NSX_EXECUTORCH_ENABLE_NS_OPS=ON;
# it fails fast at Method load on provider=arm builds.

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
    CortexMPassManager,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.exir import EdgeCompileConfig, to_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from .pass_manager_ns import NsCortexMPassManager
from .quantizer_ns import NsCortexMQuantizer

# Aten ops that the NS flow tries to accelerate. Anything from this set left
# in the final graph fell back to the portable path (a qualifier failed) and
# must be provided via NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST.
_NS_CANDIDATE_ATEN_OPS = {
    "aten::sub.out",
    "aten::hardswish.out",
    "aten::mean.out",
    "aten::relu.out",
    "aten::hardtanh.out",
    "aten::clamp.out",
    "aten::leaky_relu.out",
}

_EDGE_TO_PORTABLE = {
    "aten::sub.Tensor": "aten::sub.out",
    "aten::hardswish.default": "aten::hardswish.out",
    "aten::hardswish_.default": "aten::hardswish.out",
    "aten::mean.dim": "aten::mean.out",
    "aten::relu.default": "aten::relu.out",
    "aten::relu_.default": "aten::relu.out",
    "aten::hardtanh.default": "aten::hardtanh.out",
    "aten::hardtanh_.default": "aten::hardtanh.out",
    "aten::clamp.default": "aten::clamp.out",
    "aten::clamp_.default": "aten::clamp.out",
    "aten::leaky_relu.default": "aten::leaky_relu.out",
    "aten::leaky_relu_.default": "aten::leaky_relu.out",
}


@dataclass
class ExportResult:
    executorch_program: Any
    edge_ops: dict[str, int]
    portable_fallback_ops: list[str] = field(default_factory=list)
    # Edge-stage graph module (functional cortex_m/cortex_m_ns ops with
    # Python reference impls); usable for host-side numeric checks. The
    # serialized program uses .out variants which have no host kernels.
    edge_module: Any = None
    # Provider this program was lowered for; recorded in the sidecar.
    kernel_provider: str = "arm"
    # Method I/O quantization parameters when exported with int8_io=True:
    # {"inputs": {idx: (scale, zp, qmin, qmax, dtype)}, "outputs": {...}}.
    # Empty when the method keeps the float32 boundary.
    io_qparams: dict = field(default_factory=dict)

    @property
    def portable_select_ops_list(self) -> str:
        """Value for the NSX_EXECUTORCH_PORTABLE_SELECT_OPS_LIST cache option
        covering the aten fallbacks left in this program."""
        return ",".join(sorted(self.portable_fallback_ops))

    def write_pte(self, path, sidecar: bool = True) -> None:
        """Serialize the program; by default also write the `<path>.json`
        sidecar manifest that target builds use to self-configure."""
        with open(path, "wb") as f:
            f.write(self.executorch_program.buffer)
        if sidecar:
            from .manifest import write_sidecar

            write_sidecar(self, path, self.kernel_provider)


def _stock_edge_compile_config() -> EdgeCompileConfig:
    # Must match backends/cortex_m/test/tester.py CortexMToEdge exactly.
    return EdgeCompileConfig(
        preserve_ops=[
            torch.ops.aten.linear.default,
            torch.ops.aten.hardsigmoid.default,
            torch.ops.aten.hardsigmoid_.default,
            torch.ops.aten.hardswish.default,
            torch.ops.aten.hardswish_.default,
        ],
        _check_ir_validity=False,
        _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
    )


def _count_edge_ops(exported_program) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "_name", None) or str(node.target)
        counts[name] = counts.get(name, 0) + 1
    return counts


def is_pt2e_quantized(exported_program: torch.export.ExportedProgram) -> bool:
    """True when the graph already carries PT2E quantize/dequantize ops."""
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "_name", None) or str(node.target)
        if "quantized_decomposed" in name or "torchao.quant" in name:
            return True
    return False


def _check_kernel_provider(kernel_provider: str) -> None:
    if kernel_provider not in ("arm", "ns"):
        raise ValueError(
            f"kernel_provider must be 'arm' or 'ns', got {kernel_provider!r}"
        )


def _as_exported_program(
    model: "torch.nn.Module | torch.export.ExportedProgram",
    example_inputs: tuple[Any, ...],
) -> torch.export.ExportedProgram:
    if isinstance(model, torch.export.ExportedProgram):
        return model
    try:
        model = model.eval()
    except NotImplementedError:
        # torch.export.ExportedProgram.module() graph modules refuse eval();
        # they are already in inference form.
        pass
    return torch.export.export(model, example_inputs, strict=True)


def quantize(
    model: "torch.nn.Module | torch.export.ExportedProgram",
    example_inputs: tuple[Any, ...],
    kernel_provider: str = "arm",
    calibration_samples: Optional[Sequence[tuple[Any, ...]]] = None,
) -> torch.export.ExportedProgram:
    """PT2E-quantize a float model with the provider's quantizer.

    This is THE quantization step for the nsx-executorch flow — the CLI,
    export(), and fixture tooling all funnel through it. `model` may be an
    eager module or a float ExportedProgram; an already-quantized program is
    rejected (there is nothing left to do — pass it to export() instead).
    Calibration falls back to `example_inputs` when no samples are given.
    Returns the quantized ExportedProgram (int8 weights/activations,
    quantize/dequantize at the float method boundary) — the artifact to
    hand to torch.export.save() or export().
    """
    _check_kernel_provider(kernel_provider)
    quantizer = NsCortexMQuantizer() if kernel_provider == "ns" else CortexMQuantizer()

    exported = _as_exported_program(model, example_inputs)
    if is_pt2e_quantized(exported):
        raise ValueError(
            "the model is already PT2E-quantized; there is nothing to "
            "quantize — pass it directly to export()"
        )

    prepared = prepare_pt2e(exported.module(), quantizer)
    if calibration_samples is not None:
        for sample in calibration_samples:
            prepared(*sample)
    else:
        prepared(*example_inputs)
    converted = convert_pt2e(prepared)
    return torch.export.export(converted, example_inputs, strict=True)


def export(
    model: "torch.nn.Module | torch.export.ExportedProgram",
    example_inputs: tuple[Any, ...],
    kernel_provider: str = "arm",
    calibration_samples: Optional[Sequence[tuple[Any, ...]]] = None,
    target_config: Optional[CortexMTargetConfig] = None,
    int8_io: bool = False,
) -> ExportResult:
    """Quantize (if needed), lower and serialize `model` for the provider.

    `model` may be an eager module or a `torch.export.ExportedProgram` (e.g.
    a loaded .pt2). A float program is PT2E-quantized via quantize() with
    the provider's quantizer; a program that already carries PT2E
    quantize/dequantize ops skips quantization and goes straight to kernel
    lowering, so its baked-in quantization recipe decides what the pass
    managers can match. `calibration_samples` is only meaningful when
    quantization happens here and is rejected otherwise.

    With `int8_io=True` the serialized method takes and returns the int8
    tensors directly (like an int8 TFLite model): the boundary
    quantize/dequantize ops are removed via ExecuTorch's QuantizeInputs/
    QuantizeOutputs passes and their scales/zero-points are reported in
    `ExportResult.io_qparams` — the caller is then responsible for
    quantizing inputs and dequantizing outputs on the host side. The
    default keeps the float32 method boundary.

    Returns an ExportResult with the ExecutorchProgramManager, the edge op
    histogram and the list of NS-candidate aten ops that stayed on the
    portable fallback path (with the matching select-ops list value).
    """
    _check_kernel_provider(kernel_provider)
    pass_manager_cls = NsCortexMPassManager if kernel_provider == "ns" else CortexMPassManager

    exported = _as_exported_program(model, example_inputs)
    if is_pt2e_quantized(exported):
        if calibration_samples is not None:
            raise ValueError(
                "calibration_samples were given, but the model is already "
                "PT2E-quantized; calibration only applies when this export "
                "performs the quantization"
            )
        final_export = exported
    else:
        final_export = quantize(
            exported,
            example_inputs,
            kernel_provider=kernel_provider,
            calibration_samples=calibration_samples,
        )
    edge_manager = to_edge(final_export, compile_config=_stock_edge_compile_config())

    io_qparams: dict = {}
    if int8_io:
        # Must run while the boundary ops are still quantized_decomposed
        # (the Cortex-M pass manager below rewrites them to cortex_m::).
        # edge_program_manager=None keeps the qparams out of the PTE's
        # config methods; they are reported via ExportResult instead.
        from executorch.exir.passes.quantize_io_pass import (
            QuantizeInputs,
            QuantizeOutputs,
        )

        io_program = edge_manager.exported_program()
        quantize_inputs = QuantizeInputs(
            None,
            list(range(len(io_program.graph_signature.user_inputs))),
            exported_program=io_program,
        )
        quantize_outputs = QuantizeOutputs(
            None,
            list(range(len(io_program.graph_signature.user_outputs))),
            exported_program=io_program,
        )
        quantize_inputs.call(io_program.graph_module)
        quantize_outputs.call(io_program.graph_module)
        io_qparams = {
            "inputs": quantize_inputs.quant_args,
            "outputs": quantize_outputs.dequant_args,
        }

    pass_manager = pass_manager_cls(
        edge_manager.exported_program(), target_config=target_config
    )
    edge_manager._edge_programs["forward"] = pass_manager.transform()

    edge_ops = _count_edge_ops(edge_manager.exported_program())
    fallbacks = sorted(
        {_EDGE_TO_PORTABLE[name] for name in edge_ops if name in _EDGE_TO_PORTABLE}
    )
    edge_module = edge_manager.exported_program().module()

    executorch_program = edge_manager.to_executorch()
    return ExportResult(
        executorch_program=executorch_program,
        edge_ops=edge_ops,
        portable_fallback_ops=fallbacks,
        edge_module=edge_module,
        kernel_provider=kernel_provider,
        io_qparams=io_qparams,
    )
