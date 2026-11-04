# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# HeliaAotPartitioner: claims the parts of a helia-torch (ns profile) edge
# program that heliaAOT can compile.
#
# It runs on the quantized edge program before the NS passes, where every int8
# operator is still dequantize -> aten op -> quantize. heliaAOT's PT2E front end
# (helia_aot.converters.pt2e) decides support, one operator group (the
# dequantizes, the operator, a fused ReLU and the output quantize) at a time,
# so a claimed partition always converts. The float<->int8 conversions at the
# int8 graph's edges stay in ExecuTorch: every delegate takes and returns int8
# tensors.

from __future__ import annotations

import logging

import torch
from executorch.exir.backend.partitioner import DelegationSpec, Partitioner, PartitionResult
from executorch.exir.backend.utils import tag_constant_data
from helia_aot.converters.pt2e import analyze_program
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

from . import backend as _backend  # noqa: F401 - defines HeliaAotBackend, which lowering looks up by name
from .blob import BACKEND_ID
from .codegen import HeliaAotOptions, compile_specs_for

logger = logging.getLogger(__name__)

_DEQUANTIZE = ("quantized_decomposed::dequantize_per_tensor", "quantized_decomposed::dequantize_per_channel")


def duplicate_dequantizes(graph_module: torch.fx.GraphModule) -> int:
    """Give every consumer of a shared dequantize its own copy, so a delegated operator and one that stays in
    ExecuTorch never share a float value across the partition edge. Returns the number of copies made.

    Run it on the edge program before ``to_backend(HeliaAotPartitioner(...))``, as ``export()`` does: ExecuTorch
    refuses a partitioner that changes the graph, so the partitioner cannot do it itself.
    """
    graph = graph_module.graph
    copies = 0
    for node in list(graph.nodes):
        schema = getattr(node.target, "_schema", None)
        if node.op != "call_function" or schema is None or schema.name not in _DEQUANTIZE:
            continue
        for user in list(node.users)[1:]:
            with graph.inserting_before(user):
                copy = graph.node_copy(node)
            user.replace_input_with(node, copy)
            copies += 1
    if copies:
        graph.lint()
        graph_module.recompile()
    return copies


def merge_dequantizes(graph_module: torch.fx.GraphModule) -> int:
    """Undo ``duplicate_dequantizes`` where it was not needed: dequantizes of one value with the same parameters
    that are still in the graph after partitioning become one again. Returns the number of nodes removed.

    A copy that went into a delegate is gone from the graph, so only consumers that all stayed in ExecuTorch
    share a dequantize again, and the program runs no more dequantize kernels than without the delegate.
    """
    graph = graph_module.graph
    first: dict[tuple, torch.fx.Node] = {}
    removed = 0
    for node in list(graph.nodes):
        schema = getattr(node.target, "_schema", None)
        if node.op != "call_function" or schema is None or schema.name not in _DEQUANTIZE:
            continue
        key = (node.target, tuple(node.args), tuple(sorted(node.kwargs.items())))
        kept = first.setdefault(key, node)
        if kept is not node:
            node.replace_all_uses_with(kept)
            graph.erase_node(node)
            removed += 1
    if removed:
        graph.lint()
        graph_module.recompile()
    return removed


class _Supported(OperatorSupportBase):
    def __init__(self, nodes: set[str]):
        self.nodes = nodes

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.name in self.nodes


class HeliaAotPartitioner(Partitioner):
    """Delegate the int8 subgraphs heliaAOT's PT2E front end converts to heliaAOT-generated code."""

    def __init__(self, options: HeliaAotOptions, source):
        """``source`` is the saved program file (``save_program``): each delegated partition is a part of it."""
        super().__init__()
        self.options = options
        self.source = source
        self.rejected: dict[str, str] = {}

    def partition(self, exported_program) -> PartitionResult:
        gm = exported_program.graph_module
        support = analyze_program(exported_program)
        self.rejected = dict(support.rejected)
        partitions = CapabilityBasedPartitioner(
            gm, _Supported(support.supported_nodes), allows_single_node_partition=True
        ).propose_partitions()
        tags: dict[str, DelegationSpec] = {}
        for partition in partitions:
            names = {n.name for n in partition.nodes}
            groups = [g for g in support.groups if g.head in names]
            if any(not g.nodes <= names for g in groups):
                # The capability partitioner split an operator group (to avoid a cycle): keep it in ExecuTorch.
                logger.info("helia-aot: partition %s splits an operator group; not delegated", partition.id)
                continue
            if not any(g.emits_operator for g in groups):
                continue  # only aliases (zero pads): nothing to compile
            tag = f"helia_aot_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
            # The partition's index among the delegated ones names it in the program file (its plan's subgraph).
            tags[tag] = DelegationSpec(BACKEND_ID, compile_specs_for(self.options, self.source, len(tags)))
        tag_constant_data(exported_program)
        for name, reason in self.rejected.items():
            logger.info("helia-aot: %s stays in ExecuTorch: %s", name, reason)
        return PartitionResult(tagged_exported_program=exported_program, partition_tags=tags)
