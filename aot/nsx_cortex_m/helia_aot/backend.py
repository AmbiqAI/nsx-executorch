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

import torch
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from helia_aot.converters.pt2e import import_program

from .blob import SCALAR_TYPES, DelegateBlob, TensorContract
from .codegen import generate_module, module_name_for, options_from_specs

logger = logging.getLogger(__name__)


def _contract(node) -> TensorContract:
    val = node.meta["val"]
    dtype = val.dtype
    name = str(dtype).replace("torch.", "")
    if name not in SCALAR_TYPES:
        raise NotImplementedError(f"delegate tensor dtype {dtype}")
    return TensorContract(SCALAR_TYPES[name], val.numel() * torch.empty((), dtype=dtype).element_size())


class HeliaAotBackend(BackendDetails):
    @staticmethod
    def preprocess(edge_program, compile_specs) -> PreprocessResult:
        options = options_from_specs(list(compile_specs))
        air = import_program(edge_program, name="partition").model
        name = module_name_for(air, options.module_prefix)
        module_dir = generate_module(air, options, name)

        graph = edge_program.graph_module.graph
        user_inputs = {s.arg.name for s in edge_program.graph_signature.input_specs if s.kind.name == "USER_INPUT"}
        placeholders = [n for n in graph.nodes if n.op == "placeholder" and n.name in user_inputs]
        outputs = list(next(n for n in graph.nodes if n.op == "output").args[0])
        blob = DelegateBlob(
            module=name,
            inputs=tuple(_contract(p) for p in placeholders),
            outputs=tuple(_contract(o) for o in outputs),
        )
        logger.info("helia-aot: partition -> %s (%d AIR operators) at %s", name, len(air.operators), module_dir)
        return PreprocessResult(processed_bytes=blob.pack())
