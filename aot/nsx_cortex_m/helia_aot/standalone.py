# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Compile a whole PyTorch model with heliaAOT, without ExecuTorch.
#
# When heliaAOT covers every operator, the delegated PTE is just
# quantize -> HeliaAotBackend -> dequantize, and the ExecuTorch runtime only
# loads the program and calls the generated code. compile_model() produces that
# generated code on its own: one plain heliaAOT NSX module that the
# application calls directly (<name>_model_init / <name>_model_run), with no
# PTE, ExecuTorch runtime or delegate registry. The module computes exactly
# what the delegate computes (same frontend, same AIR, same heliaAOT rules).
#
# The method boundary becomes the application's job: inputs are passed in as
# int8 and outputs come back as int8, in the tensors' physical (dim order)
# layout; CompiledModel.inputs/outputs give the scales, zero points and
# layouts to (de)quantize with.
#
# A model with any operator heliaAOT cannot compile is refused
# (NotFullyDelegableError lists them); export(..., helia_aot=...) handles such
# models, leaving those operators in ExecuTorch.

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from .codegen import HeliaAotOptions, generate_module, module_name_for
from .frontend import (
    UnsupportedNodeError,
    dim_order,
    edge_to_air,
    lower_for_frontend,
    physical_shape,
    retype_quantized_outputs,
    target_name,
    unsupported_reason,
)

_Q = "cortex_m.quantize_per_tensor"
_DQ = "cortex_m.dequantize_per_tensor"
MANIFEST_FORMAT = 1


class NotFullyDelegableError(UnsupportedNodeError):
    """The model has operators heliaAOT cannot compile; `rejected` maps each node to the reason."""

    def __init__(self, rejected: dict[str, str]):
        self.rejected = dict(rejected)
        lines = "\n".join(f"  {name}: {reason}" for name, reason in self.rejected.items())
        super().__init__(
            f"{len(self.rejected)} operator(s) cannot be compiled by heliaAOT, so the model needs ExecuTorch "
            f"(export(..., helia_aot=HeliaAotOptions(...)) delegates the rest):\n{lines}"
        )


@dataclass(frozen=True)
class IoTensor:
    """One model input or output at the int8 boundary.

    shape is the logical (PyTorch) shape; physical_shape is the byte layout the generated code uses, i.e.
    shape permuted by dim_order (NHWC for a channels_last 4-D tensor). real = (q - zero_point) * scale.
    """

    name: str
    shape: tuple[int, ...]
    dim_order: tuple[int, ...]
    physical_shape: tuple[int, ...]
    dtype: str
    scale: float
    zero_point: int
    qmin: int
    qmax: int

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """`x` (logical shape, float) as the int8 tensor in physical order the module reads."""
        q = torch.clamp(torch.round(x / self.scale) + self.zero_point, self.qmin, self.qmax).to(torch.int8)
        return q.permute(self.dim_order).contiguous()

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        """The int8 module output `q` (physical order) as a float tensor in logical order."""
        inverse = [self.dim_order.index(d) for d in range(len(self.dim_order))]
        q = q.reshape(self.physical_shape).permute(inverse)
        return (q.to(torch.float32) - self.zero_point) * self.scale


@dataclass
class AirProgram:
    """A model converted to heliaAOT AIR, with its int8 boundary."""

    air: Any  # helia_aot.air.AirModel
    inputs: list[IoTensor]
    outputs: list[IoTensor]


@dataclass
class CompiledModel:
    name: str
    module_dir: Path
    inputs: list[IoTensor]
    outputs: list[IoTensor]
    air: Any = field(repr=False, default=None)

    def manifest(self) -> dict:
        return {
            "format": MANIFEST_FORMAT,
            "module": self.name,
            "inputs": [asdict(t) for t in self.inputs],
            "outputs": [asdict(t) for t in self.outputs],
        }

    def write_manifest(self, path: Optional[Path] = None) -> Path:
        """Write the I/O manifest (default <module_dir>/<name>_io.json) and return its path."""
        path = Path(path) if path is not None else self.module_dir / f"{self.name}_io.json"
        path.write_text(json.dumps(self.manifest(), indent=2) + "\n")
        return path


def _io_tensor(placeholder_or_output: torch.fx.Node, qnode: torch.fx.Node, name: str) -> IoTensor:
    val = placeholder_or_output.meta["val"]
    _, scale, zp, qmin, qmax, dtype = qnode.args[:6]
    return IoTensor(
        name=name,
        shape=tuple(int(d) for d in val.shape),
        dim_order=tuple(dim_order(val)),
        physical_shape=tuple(physical_shape(val)),
        dtype=str(dtype).replace("torch.", ""),
        scale=float(scale),
        zero_point=int(zp),
        qmin=int(qmin),
        qmax=int(qmax),
    )


def _boundary(ep) -> tuple[list[IoTensor], list[IoTensor], set[str]]:
    """The int8 boundary of a frontend-form program and the names of its boundary nodes."""
    graph = ep.graph_module.graph
    user_inputs = [s.arg.name for s in ep.graph_signature.input_specs if s.kind.name == "USER_INPUT"]
    by_name = {n.name: n for n in graph.nodes}
    inputs, boundary = [], set()
    rejected: dict[str, str] = {}
    for name in user_inputs:
        users = list(by_name[name].users)
        if len(users) != 1 or target_name(users[0]) != _Q:
            rejected[name] = "model input is not quantized exactly once to int8"
            continue
        inputs.append(_io_tensor(by_name[name], users[0], name))
        boundary.add(users[0].name)
    outputs = []
    out = next(n for n in graph.nodes if n.op == "output")
    for i, arg in enumerate(out.args[0]):
        if not isinstance(arg, torch.fx.Node) or target_name(arg) != _DQ:
            rejected[getattr(arg, "name", f"output{i}")] = "model output is not dequantized from int8"
            continue
        if any(u.op != "output" for u in arg.users):
            rejected[arg.name] = "output dequantize also feeds other operators"
            continue
        outputs.append(_io_tensor(arg.args[0], arg, f"output{i}"))
        boundary.add(arg.name)
    if rejected:
        raise NotFullyDelegableError(rejected)
    return inputs, outputs, boundary


def _edge_program(model, example_inputs, kernel_provider, calibration_samples):
    from executorch.exir import to_edge

    from ..export import _as_exported_program, _stock_edge_compile_config, is_pt2e_quantized, quantize

    if isinstance(model, torch.export.ExportedProgram) and is_pt2e_quantized(model):
        if calibration_samples is not None:
            raise ValueError("calibration_samples were given, but the model is already PT2E-quantized")
        quantized = model
    else:
        if example_inputs is None:
            raise ValueError("example_inputs are needed to quantize a float model")
        quantized = quantize(_as_exported_program(model, example_inputs), example_inputs,
                             kernel_provider=kernel_provider, calibration_samples=calibration_samples)
    return to_edge(quantized, compile_config=_stock_edge_compile_config()).exported_program()


def program_to_air(
    model: "torch.nn.Module | torch.export.ExportedProgram",
    example_inputs: Optional[tuple[Any, ...]] = None,
    *,
    calibration_samples: Optional[Sequence[tuple[Any, ...]]] = None,
    target_config=None,
    name: str = "model",
) -> AirProgram:
    """Convert a whole model to heliaAOT AIR (the step before code generation).

    `model` may be an eager module or float ExportedProgram (quantized here with the ns quantizer, which
    needs example_inputs) or an already PT2E-quantized ExportedProgram (e.g. torch.export.load() of the
    program helia-torch's quantize() returns). Raises NotFullyDelegableError if any operator would have to
    stay in ExecuTorch.
    """
    ep = lower_for_frontend(_edge_program(model, example_inputs, "ns", calibration_samples), target_config)
    retype_quantized_outputs(ep)
    inputs, outputs, boundary = _boundary(ep)
    rejected = {
        n.name: reason
        for n in ep.graph_module.graph.nodes
        if n.op == "call_function" and n.name not in boundary
        and (reason := unsupported_reason(ep, n)) is not None
    }
    if rejected:
        raise NotFullyDelegableError(rejected)
    try:
        air = edge_to_air(ep, name=name)
    except UnsupportedNodeError as exc:
        if isinstance(exc, NotFullyDelegableError):
            raise
        raise NotFullyDelegableError({"<graph>": str(exc)}) from exc
    return AirProgram(air=air, inputs=inputs, outputs=outputs)


def compile_model(
    model: "torch.nn.Module | torch.export.ExportedProgram",
    example_inputs: Optional[tuple[Any, ...]],
    options: HeliaAotOptions,
    *,
    name: Optional[str] = None,
    calibration_samples: Optional[Sequence[tuple[Any, ...]]] = None,
    target_config=None,
) -> CompiledModel:
    """Compile a whole model to one heliaAOT NSX module, with no ExecuTorch runtime.

    Writes <options.output_dir>/<name>/ (a heliaAOT NSX module: call <name>_model_init() once, then fill
    the context's input tensors and call <name>_model_run()) plus <name>_io.json, the int8 boundary. `name`
    must be a C identifier; by default it is <options.module_prefix>_<hash of the AIR>.
    """
    if name is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"module name {name!r} is not a C identifier")
    program = program_to_air(model, example_inputs, calibration_samples=calibration_samples,
                             target_config=target_config, name=name or "model")
    name = name or module_name_for(program.air, options.module_prefix)
    module_dir = generate_module(program.air, options, name, executorch_registration=False)
    compiled = CompiledModel(name=name, module_dir=module_dir, inputs=program.inputs, outputs=program.outputs,
                             air=program.air)
    compiled.write_manifest()
    return compiled
