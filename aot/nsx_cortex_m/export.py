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


def export(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    kernel_provider: str = "arm",
    calibration_samples: Optional[Sequence[tuple[Any, ...]]] = None,
    target_config: Optional[CortexMTargetConfig] = None,
) -> ExportResult:
    """Quantize, lower and serialize `model` for the selected kernel provider.

    Returns an ExportResult with the ExecutorchProgramManager, the edge op
    histogram and the list of NS-candidate aten ops that stayed on the
    portable fallback path (with the matching select-ops list value).
    """
    if kernel_provider not in ("arm", "ns"):
        raise ValueError(
            f"kernel_provider must be 'arm' or 'ns', got {kernel_provider!r}"
        )

    if kernel_provider == "ns":
        quantizer = NsCortexMQuantizer()
        pass_manager_cls = NsCortexMPassManager
    else:
        quantizer = CortexMQuantizer()
        pass_manager_cls = CortexMPassManager

    try:
        model = model.eval()
    except NotImplementedError:
        # torch.export.ExportedProgram.module() graph modules refuse eval();
        # they are already in inference form.
        pass
    exported = torch.export.export(model, example_inputs, strict=True)
    graph_module = exported.module()

    prepared = prepare_pt2e(graph_module, quantizer)
    if calibration_samples is not None:
        for sample in calibration_samples:
            prepared(*sample)
    else:
        prepared(*example_inputs)
    converted = convert_pt2e(prepared)

    final_export = torch.export.export(converted, example_inputs, strict=True)
    edge_manager = to_edge(final_export, compile_config=_stock_edge_compile_config())

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
    )
