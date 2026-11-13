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
from .standalone import CompiledModel, NotFullyDelegableError, _quantized_program, compile_model

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
    operators that kept an automatic build off the module path, and is empty otherwise; `module_error` is
    heliaAOT's error when it covers every operator but could not compile the model as one module.
    """

    mode: str
    compiled: Optional[CompiledModel] = None
    export_result: Optional["ExportResult"] = None
    rejected: dict[str, str] = field(default_factory=dict)
    module_error: str = ""


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

    mode="auto" (the default) uses compile_model() when heliaAOT covers every operator and falls back to
    export(..., helia_aot=options) when it does not; "module" and "delegate" force one path. A float model is
    quantized once (ns quantizer, `calibration_samples`), so both paths see the same program. `name` names the
    plain module; `int8_io` applies to the delegate's PTE (a plain module's boundary is always int8).
    `test_inputs` give a plain module heliaAOT's test case (see compile_model); a delegate build has none.

    In "auto", a model heliaAOT covers but cannot compile as one module (its HeliaAotError is kept in
    `module_error`) is also tried as a delegate.

    Raises NotFullyDelegableError when mode="module" and the model has operators heliaAOT cannot compile, and
    heliaAOT's own error when it cannot compile what the chosen path gives it.
    """
    from helia_aot.converters.pt2e import UnsupportedProgramError
    from helia_aot.errors import HeliaAotError

    from ..export import export

    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    program = _quantized_program(model, example_inputs, "ns", calibration_samples)

    rejected: dict[str, str] = {}
    module_error = ""
    if mode != DELEGATE:
        try:
            compiled = compile_model(program, example_inputs, options, name=name, test_inputs=test_inputs)
        except NotFullyDelegableError as exc:
            if mode == MODULE:
                raise
            rejected = exc.rejected
        except UnsupportedProgramError:
            raise  # the program cannot be imported at all: the delegate would refuse it too
        except HeliaAotError as exc:
            # heliaAOT covers every operator but could not compile the whole model (it does not fit the
            # platform's memory, say). Its partitions may still compile, so an automatic build tries them.
            if mode == MODULE:
                raise
            module_error = str(exc)
        else:
            return HeliaAotBuild(mode=MODULE, compiled=compiled)

    result = export(program, example_inputs, kernel_provider="ns", target_config=target_config, int8_io=int8_io,
                    helia_aot=options)
    return HeliaAotBuild(mode=DELEGATE, export_result=result, rejected=rejected, module_error=module_error)
