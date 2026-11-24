# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Pick how heliaAOT runs a model: as a plain module or as an ExecuTorch delegate.
#
#   build(model, inputs, HeliaAotOptions(output_dir=...))
#
# compiles a model heliaAOT fully covers with compile_model() (one plain NSX module, no ExecuTorch
# runtime: the smallest and fastest build), and anything else with export(..., helia_aot=...) (a PTE in
# which heliaAOT runs the subgraphs it supports and ExecuTorch the rest). mode="module" or "delegate"
# forces one path; forcing "module" on a model heliaAOT does not fully cover raises NotFullyDelegableError.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .codegen import HeliaAotOptions
from .standalone import CompiledModel, NotFullyDelegableError, _compile_quantized, _quantized_program, check_module_name

if TYPE_CHECKING:
    import torch

    from ..export import ExportResult

AUTO = "auto"
MODULE = "module"
DELEGATE = "delegate"
MODES = (AUTO, MODULE, DELEGATE)


@dataclass
class HeliaAotBuild:
    """What build() produced.

    mode is the path taken, "module" or "delegate". A module build has `compiled` (the NSX module and its
    int8 I/O contract); a delegate build has `export_result` (write it with write_pte()). `rejected` names the
    nodes (operators, or inputs and outputs) that kept an automatic build off the module path, and is empty
    otherwise.
    """

    mode: str
    compiled: Optional[CompiledModel] = None
    export_result: Optional["ExportResult"] = None
    rejected: dict[str, str] = field(default_factory=dict)


def build(
    model: "torch.nn.Module | torch.export.ExportedProgram",
    example_inputs: tuple[Any, ...],
    options: HeliaAotOptions,
    *,
    mode: str = AUTO,
    name: Optional[str] = None,
    calibration_samples: Optional[Sequence[tuple[Any, ...]]] = None,
    target_config=None,
    int8_io: bool = False,
    test_inputs: Optional[Sequence[Any]] = None,
) -> HeliaAotBuild:
    """Compile `model` with heliaAOT, as a plain module when it can and through ExecuTorch otherwise.

    mode="auto" (the default) uses compile_model() when heliaAOT covers the whole model, inputs and outputs
    included, and falls back to export(..., helia_aot=options) when it does not; "module" and "delegate" force
    one path. A float model is quantized once (ns quantizer, `calibration_samples`) and lowered to the edge
    dialect once, so both paths see the same program. `name` names the plain module; `int8_io` applies to the
    delegate's PTE (a plain module's boundary is always int8).
    `test_inputs` give a plain module heliaAOT's test case (see compile_model); a delegate build has none.

    Raises NotFullyDelegableError when mode="module" and heliaAOT cannot compile the whole model, ValueError
    when `name` is not a C identifier, and heliaAOT's own error when it cannot compile what the chosen path gives
    it. A model heliaAOT covers but cannot compile as one module (it does not fit the platform's memory, say)
    raises that error in "auto" too: as a delegate it would be one partition, the same model.
    """
    from executorch.exir import to_edge

    from ..export import _lower_edge, _stock_edge_compile_config

    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    check_module_name(name)
    program = _quantized_program(model, example_inputs, "ns", calibration_samples)
    edge_manager = to_edge(program, compile_config=_stock_edge_compile_config())

    rejected: dict[str, str] = {}
    if mode != DELEGATE:
        try:
            compiled = _compile_quantized(program, edge_manager.exported_program(), options, name, test_inputs)
        except NotFullyDelegableError as exc:
            if mode == MODULE:
                raise
            rejected = exc.rejected
        else:
            return HeliaAotBuild(mode=MODULE, compiled=compiled)

    result = _lower_edge(program, edge_manager, "ns", target_config, int8_io, options)
    return HeliaAotBuild(mode=DELEGATE, export_result=result, rejected=rejected)
