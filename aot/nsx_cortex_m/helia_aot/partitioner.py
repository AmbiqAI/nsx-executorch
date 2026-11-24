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
# dequantizes, the operator, a fused ReLU and the output quantize) at a time.
# That analysis does not judge a partition's inputs and outputs (two outputs
# that are one int8 tensor, say), so each proposed partition is also cut and
# imported as HeliaAotBackend will import it; the groups it refuses stay in
# ExecuTorch, and a claimed partition always converts. The float<->int8
# conversions at the int8 graph's edges stay in ExecuTorch: every delegate
# takes and returns int8 tensors.

from __future__ import annotations

import copy
import logging

import torch
from executorch.exir.backend.partitioner import DelegationSpec, Partitioner, PartitionResult
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.lowered_backend_module import (
    create_exported_program_from_submodule,
    create_submodule_from_nodes,
)
from helia_aot.converters.pt2e import analyze_program, import_program
from helia_aot.errors import UnsupportedProgramError
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

    def __init__(self, options: HeliaAotOptions, source, program):
        """``source`` is the saved program file (``save_program``): each delegated partition is a part of it.
        ``program`` is the edge program ``to_backend()`` lowers: ExecuTorch gives ``partition()`` a copy with fake
        weights, and a partition is tried with the real ones."""
        super().__init__()
        self.options = options
        self.source = source
        self.program = program
        self.rejected: dict[str, str] = {}

    def partition(self, exported_program) -> PartitionResult:
        support = analyze_program(exported_program)
        self.rejected = dict(support.rejected)
        groups = list(support.groups)
        while True:
            partitions = self._propose(exported_program.graph_module, groups)
            refused = _refused_partitions(exported_program, partitions, self.program)
            if not refused:
                break
            # Keep the refused groups in ExecuTorch and partition again: a smaller partition has other edges.
            for index, rejected in refused.items():
                self.rejected.update(rejected)
                names = partitions[index]
                dropped = [g for g in groups if g.head in names and g.nodes & rejected.keys()]
                for group in dropped or [g for g in groups if g.head in names]:
                    groups.remove(group)
        tags: dict[str, DelegationSpec] = {}
        nodes = {n.name: n for n in exported_program.graph_module.graph.nodes}
        for names in partitions:
            tag = f"helia_aot_{len(tags)}"
            for name in names:
                nodes[name].meta["delegation_tag"] = tag
            # The partition's index among the delegated ones names it in the program file (its plan's subgraph).
            tags[tag] = DelegationSpec(BACKEND_ID, compile_specs_for(self.options, self.source, len(tags)))
        tag_constant_data(exported_program)
        for name, reason in self.rejected.items():
            logger.info("helia-aot: %s stays in ExecuTorch: %s", name, reason)
        return PartitionResult(tagged_exported_program=exported_program, partition_tags=tags)

    @staticmethod
    def _propose(graph_module: torch.fx.GraphModule, groups: list) -> list[set[str]]:
        """The node names of each partition worth delegating, made of whole operator groups."""
        supported = {name for group in groups for name in group.nodes}
        proposed = CapabilityBasedPartitioner(
            graph_module, _Supported(supported), allows_single_node_partition=True
        ).propose_partitions()
        partitions = []
        for partition in proposed:
            names = {n.name for n in partition.nodes}
            members = [g for g in groups if g.head in names]
            if any(not g.nodes <= names for g in members):
                # The capability partitioner split an operator group (to avoid a cycle): keep it in ExecuTorch.
                logger.info("helia-aot: partition %s splits an operator group; not delegated", partition.id)
                continue
            if not any(g.emits_operator for g in members):
                continue  # only aliases (zero pads): nothing to compile
            partitions.append(names)
        # In graph order (the capability partitioner proposes them last first), so partition 0 runs first.
        order = {node.name: index for index, node in enumerate(graph_module.graph.nodes)}
        return sorted(partitions, key=lambda names: min(order[name] for name in names))


def _refused_partitions(exported_program, partitions: list[set[str]], weights) -> dict[int, dict[str, str]]:
    """Cut each partition out of a copy of the program, as ``to_backend()`` will, and import it as
    ``HeliaAotBackend.preprocess()`` will. Returns what heliaAOT refuses, by partition index.

    ``weights`` is the program with real weights (``exported_program`` may hold fake ones); the copy takes them
    by name, without copying them: cutting partitions changes the graph and signature only.
    """
    tensors = (*exported_program.state_dict.values(), *exported_program.constants.values())
    trial = copy.deepcopy(exported_program, {id(t): t for t in tensors})
    trial.state_dict.update({k: v for k, v in weights.state_dict.items() if k in trial.state_dict})
    trial.constants.update({k: v for k, v in weights.constants.items() if k in trial.constants})
    graph_module = trial.graph_module
    for index, names in enumerate(partitions):
        for node in graph_module.graph.nodes:
            if node.name in names:
                node.meta["delegation_tag"] = f"helia_aot_trial_{index}"
    tag_constant_data(trial)  # a partition's weights go with it, as to_backend() takes them
    refused: dict[int, dict[str, str]] = {}
    for index in range(len(partitions)):
        tag = f"helia_aot_trial_{index}"
        nodes = [n for n in graph_module.graph.nodes if n.meta.get("delegation_tag") == tag]
        with graph_module._set_replace_hook(trial.graph_signature.get_replace_hook()):
            submodule, call_module = create_submodule_from_nodes(graph_module, nodes, tag)
        submodule.graph.output_node().meta = graph_module.graph.output_node().meta
        program, _, _ = create_exported_program_from_submodule(submodule, trial, tag, call_module, False)
        try:
            import_program(program, name="partition")
        except UnsupportedProgramError as err:
            refused[index] = dict(err.rejected) or {tag: str(err)}
    return refused
