# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# HeliaAotPartitioner: claims the parts of a helia-torch (ns profile) edge
# program that heliaAOT can compile.
#
# It runs on the frontend's input form: the program after the NS pass list up
# to ConvertToCortexMPass (frontend.lower_for_frontend). Support is decided by
# the frontend itself (frontend.unsupported_reason), so a claimed node always
# converts. Float<->int8 conversions at partition edges stay ExecuTorch
# cortex_m kernels: every delegate takes and returns int8 tensors.

from __future__ import annotations

import logging
import torch
from executorch.exir.backend.partitioner import DelegationSpec, Partitioner, PartitionResult
from executorch.exir.backend.utils import tag_constant_data
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

from . import backend as _backend  # noqa: F401 - defines HeliaAotBackend, which lowering looks up by name
from .blob import BACKEND_ID
from .codegen import HeliaAotOptions, compile_specs_for
from .frontend import produces_air_operator, unsupported_reason

logger = logging.getLogger(__name__)


class _FrontendSupport(OperatorSupportBase):
    def __init__(self, ep, rejected: dict[str, str]):
        self.ep = ep
        self.rejected = rejected

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        reason = unsupported_reason(self.ep, node)
        if reason is not None and node.op == "call_function":
            self.rejected[node.name] = reason
        return reason is None


class HeliaAotPartitioner(Partitioner):
    """Delegate frontend-supported int8 subgraphs to heliaAOT-generated code."""

    def __init__(self, options: HeliaAotOptions):
        super().__init__()
        self.options = options
        self.delegation_spec = DelegationSpec(BACKEND_ID, compile_specs_for(options))
        self.rejected: dict[str, str] = {}

    def partition(self, exported_program) -> PartitionResult:
        gm = exported_program.graph_module
        support = _FrontendSupport(exported_program, self.rejected)
        partitions = CapabilityBasedPartitioner(
            gm, support, allows_single_node_partition=True
        ).propose_partitions()
        tags: dict[str, DelegationSpec] = {}
        for partition in partitions:
            if not any(produces_air_operator(n) for n in partition.nodes):
                continue  # only aliases (zero pads, requantize halves): nothing to compile
            tag = f"helia_aot_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
            tags[tag] = self.delegation_spec
        tag_constant_data(exported_program)
        for name, reason in self.rejected.items():
            logger.info("helia-aot: %s stays in ExecuTorch: %s", name, reason)
        return PartitionResult(tagged_exported_program=exported_program, partition_tags=tags)
