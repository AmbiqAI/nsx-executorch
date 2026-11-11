# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# HeliaAotBackend: ExecuTorch BackendDetails for heliaAOT.
#
# preprocess() converts a partition (int8 in, int8 out: dequantize -> aten op
# -> quantize groups, see partitioner.py) to AIR with heliaAOT's PT2E front
# end, runs heliaAOT to write an NSX module into the output_dir compile spec,
# and returns the delegate blob (blob.py). The
# firmware links the module; nsx-executorch's runtime HeliaAotBackend
# (src/helia_aot_backend.cpp) finds it by name and runs it.
#
# ExecuTorch resolves a backend by class name, so this class must be named
# exactly BACKEND_ID and be imported before lowering.

from __future__ import annotations

import logging
import math

import torch
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from helia_aot.converters.pt2e import import_program

from .blob import SCALAR_TYPES, DelegateBlob, TensorContract
from .codegen import generate_module, module_name_for, options_from_specs, source_from_specs

logger = logging.getLogger(__name__)


def _contract(node, boundary) -> TensorContract:
    """The contract of one delegate argument: the int8 tensor the module reads or writes.

    The module's buffers are the importer's int8 boundary tensors, so their size is the contract. A partition
    that takes or returns anything else (a float tensor heliaAOT would quantize itself, say) cannot be a
    delegate: the runtime copies the ExecuTorch tensor's bytes into the module's buffer as they are.
    """
    val = node.meta["val"]
    if val.dtype != torch.int8:
        raise ValueError(
            f"HeliaAotBackend: delegate argument {node.name} is {val.dtype}, not int8. A delegate takes and "
            "returns int8 tensors: lower the model with export(helia_aot=...), whose partitioner leaves the "
            "float <-> int8 conversions in ExecuTorch."
        )
    nbytes = math.prod(boundary.shape)
    if nbytes != val.numel():
        raise ValueError(
            f"HeliaAotBackend: delegate argument {node.name} holds {val.numel()} values, and the module's "
            f"tensor {boundary.name} holds {nbytes}"
        )
    return TensorContract(SCALAR_TYPES[boundary.dtype], nbytes)


class HeliaAotBackend(BackendDetails):
    @staticmethod
    def preprocess(edge_program, compile_specs) -> PreprocessResult:
        options = options_from_specs(list(compile_specs))
        source = source_from_specs(list(compile_specs))
        imported = import_program(edge_program, name="partition", source=source)
        air = imported.model
        name = module_name_for(source, options.module_prefix)
        module_dir = generate_module(air, options, name)

        graph = edge_program.graph_module.graph
        user_inputs = {s.arg.name for s in edge_program.graph_signature.input_specs if s.kind.name == "USER_INPUT"}
        placeholders = [n for n in graph.nodes if n.op == "placeholder" and n.name in user_inputs]
        outputs = list(next(n for n in graph.nodes if n.op == "output").args[0])
        blob = DelegateBlob(
            module=name,
            inputs=tuple(_contract(p, t) for p, t in zip(placeholders, imported.inputs, strict=True)),
            outputs=tuple(_contract(o, t) for o, t in zip(outputs, imported.outputs, strict=True)),
        )
        logger.info("helia-aot: partition -> %s (%d AIR operators) at %s", name, len(air.operators), module_dir)
        return PreprocessResult(processed_bytes=blob.pack())
