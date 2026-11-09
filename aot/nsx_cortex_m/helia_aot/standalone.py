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
# what the delegate computes (same PT2E front end, same AIR, same heliaAOT rules).
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
from helia_aot.converters.pt2e import BoundaryTensor, UnsupportedProgramError, import_program

from .codegen import HeliaAotOptions, generate_module, module_name_for

MANIFEST_FORMAT = 1


class NotFullyDelegableError(ValueError):
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
        """`x` (logical shape, float) as the int8 tensor in physical order the module reads.

        Computed as ExecuTorch's quantize_per_tensor does, by multiplying with the reciprocal of the scale:
        dividing by the scale rounds a few values one code differently.
        """
        q = torch.round(x * (1.0 / self.scale)) + self.zero_point
        q = torch.clamp(q, self.qmin, self.qmax).to(torch.int8)
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


def _io_tensor(t: BoundaryTensor) -> IoTensor:
    return IoTensor(
        name=t.name,
        shape=t.shape,
        dim_order=t.dim_order,
        physical_shape=t.physical_shape,
        dtype=t.dtype,
        scale=t.scale,
        zero_point=t.zero_point,
        qmin=t.qmin,
        qmax=t.qmax,
    )


def _quantized_program(model, example_inputs, kernel_provider, calibration_samples):
    from ..export import _as_exported_program, is_pt2e_quantized, quantize

    if isinstance(model, torch.export.ExportedProgram) and is_pt2e_quantized(model):
        if calibration_samples is not None:
            raise ValueError("calibration_samples were given, but the model is already PT2E-quantized")
        return model
    if example_inputs is None:
        raise ValueError("example_inputs are needed to quantize a float model")
    return quantize(_as_exported_program(model, example_inputs), example_inputs,
                    kernel_provider=kernel_provider, calibration_samples=calibration_samples)


def _edge_program(model, example_inputs, kernel_provider, calibration_samples):
    from executorch.exir import to_edge

    from ..export import _stock_edge_compile_config

    quantized = _quantized_program(model, example_inputs, kernel_provider, calibration_samples)
    return to_edge(quantized, compile_config=_stock_edge_compile_config()).exported_program()


def program_to_air(
    model: "torch.nn.Module | torch.export.ExportedProgram",
    example_inputs: Optional[tuple[Any, ...]] = None,
    *,
    calibration_samples: Optional[Sequence[tuple[Any, ...]]] = None,
    name: str = "model",
    source: Any = None,
) -> AirProgram:
    """Convert a whole model to heliaAOT AIR (the step before code generation).

    `model` may be an eager module or float ExportedProgram (quantized here with the ns quantizer, which
    needs example_inputs) or an already PT2E-quantized ExportedProgram (e.g. torch.export.load() of the
    program helia-torch's quantize() returns). heliaAOT's PT2E front end converts it. Raises
    NotFullyDelegableError if any operator would have to stay in ExecuTorch, and heliaAOT's
    UnsupportedProgramError if the program cannot be imported at all (it was traced without its weights).

    `source` is the file the quantized program was saved to (helia_aot.air.AirModelSource.of_file()), which
    heliaAOT's convert() needs: the plan names the model by it. Without it, set `air.source` before converting.
    """
    ep = _edge_program(model, example_inputs, "ns", calibration_samples)
    try:
        imported = import_program(ep, name=name, source=source)
    except UnsupportedProgramError as exc:
        if not exc.rejected:
            # Not an operator heliaAOT lacks: the program itself cannot be imported (traced without its
            # weights, say), and ExecuTorch could not delegate any of it either.
            raise
        raise NotFullyDelegableError(exc.rejected) from exc
    return AirProgram(
        air=imported.model,
        inputs=[_io_tensor(t) for t in imported.inputs],
        outputs=[_io_tensor(t) for t in imported.outputs],
    )


def compile_model(
    model: "torch.nn.Module | torch.export.ExportedProgram",
    example_inputs: Optional[tuple[Any, ...]],
    options: HeliaAotOptions,
    *,
    name: Optional[str] = None,
    calibration_samples: Optional[Sequence[tuple[Any, ...]]] = None,
) -> CompiledModel:
    """Compile a whole model to one heliaAOT NSX module, with no ExecuTorch runtime.

    Writes <options.output_dir>/<name>/ (a heliaAOT NSX module: call <name>_model_init() once, then fill
    the context's input tensors and call <name>_model_run()) plus <name>_io.json, the int8 boundary. `name`
    must be a C identifier; by default it is <options.module_prefix>_<digest>_0. The quantized program is saved
    as <options.output_dir>/<options.module_prefix>_<digest>.pt2, and heliaAOT's plan names the model by that file.
    The file is reproducible (see codegen.save_program), so compiling the same quantized model again gives the
    same name.
    """
    from .codegen import save_program

    if name is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"module name {name!r} is not a C identifier")
    quantized = _quantized_program(model, example_inputs, "ns", calibration_samples)
    program = program_to_air(quantized, name=name or "model")  # refuses a model it does not cover, writing nothing
    program.air.source = save_program(quantized, options)  # the whole program: partition 0 of its file
    name = name or module_name_for(program.air.source, options.module_prefix)
    module_dir = generate_module(program.air, options, name, executorch_registration=False)
    compiled = CompiledModel(name=name, module_dir=module_dir, inputs=program.inputs, outputs=program.outputs,
                             air=program.air)
    compiled.write_manifest()
    return compiled
