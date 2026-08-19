# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# NsCortexMQuantizer: the stock CortexMQuantizer extended with annotation
# patterns for the ns-cmsis-nn additional kernels (sub, standalone
# relu/relu6/hardtanh/clamp, leaky_relu, mean). Reuses the stock pattern
# matcher/checker machinery and INT8 configs unchanged.

from typing import cast, List, Optional

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.quantizer.arm_quantizer_utils import (
    PatternCheck,
    PatternQuantizer,
    SharedQspecQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.passes.passes_utils import is_channels_last
from executorch.backends.cortex_m.quantizer.node_finders import (
    GlobalNodeFinder,
    NodeTargetNodeFinder,
)
from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_TENSOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer.pattern_checkers import CortexMAddMulCheck
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.quantizer.quantizer_support import (
    CONV_OP_PATTERNS,
    CONV_TRANSPOSE_OP_PATTERNS,
    CORTEX_M_QUANTIZER_SUPPORT_DICT,
)
from executorch.backends.cortex_m.quantizer_reporter import QuantizerReporter
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import ComposableQuantizer, Quantizer

from .pass_manager_ns import NsCortexMPassManager


def _strip_guards_fn(model: GraphModule) -> None:
    """Remove the ``_guards_fn`` call_module node torch.export inserts for
    models with input guards (torch >= 2.12). It has no users and only runs
    runtime assertions, but ExportPass-based annotation passes reject any
    call_module node."""
    for node in list(model.graph.nodes):
        if node.op == "call_module" and node.target == "_guards_fn" and not node.users:
            model.graph.erase_node(node)
            if hasattr(model, "_guards_fn"):
                delattr(model, "_guards_fn")
    model.graph.lint()
    model.recompile()


def _is_per_tensor_int8(cls, quantization_config: QuantizationConfig) -> bool:
    is_per_tensor = PatternCheck.is_per_tensor(
        quantization_config.get_input_act_qspec()
    ) and PatternCheck.is_per_tensor(quantization_config.get_output_act_qspec())
    return is_per_tensor and cls.is_int8_activations(quantization_config)


class NsUnaryActivationCheck(PatternCheck):
    """Standalone relu/relu6/hardtanh/clamp with compile-time scalar bounds."""

    @classmethod
    def check_pattern(cls, pattern) -> bool:
        node = pattern[0]
        # Bounds (hardtanh/clamp) must be compile-time scalars.
        for arg in node.args[1:]:
            if isinstance(arg, Node) or isinstance(arg, torch.Tensor):
                return False
        return True

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: QuantizationConfig
    ) -> bool:
        return _is_per_tensor_int8(cls, quantization_config)


class NsLeakyReluCheck(PatternCheck):
    """Standalone leaky_relu with a compile-time positive slope."""

    @classmethod
    def check_pattern(cls, pattern) -> bool:
        node = pattern[0]
        alpha = node.args[1] if len(node.args) > 1 else 0.01
        return isinstance(alpha, (int, float)) and alpha > 0

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: QuantizationConfig
    ) -> bool:
        return _is_per_tensor_int8(cls, quantization_config)


class NsMeanCheck(PatternCheck):
    """aten.mean.dim restricted to what arm_mean_s8 supports: rank <= 4
    contiguous tensors with compile-time integer reduce dims."""

    @classmethod
    def check_pattern(cls, pattern) -> bool:
        node = pattern[0]
        # Layout qualifier applies to the INPUT tensor (the kernel reduces
        # over it); the output of e.g. keepdim mean is (N, C, 1, 1) which
        # is_channels_last treats as ambiguous.
        input_arg = node.args[0]
        if not isinstance(input_arg, Node):
            return False
        tensor = get_first_fake_tensor(input_arg)
        if tensor.dim() > 4 or is_channels_last(tensor):
            return False
        dims = node.args[1] if len(node.args) > 1 else None
        if isinstance(dims, int):
            dims = [dims]
        if not isinstance(dims, (list, tuple)) or not dims:
            return False
        return all(isinstance(d, int) for d in dims)

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: QuantizationConfig
    ) -> bool:
        return _is_per_tensor_int8(cls, quantization_config)


NS_EXTRA_PATTERNS = {
    (torch.ops.aten.sub.Tensor,): CortexMAddMulCheck,
    (torch.ops.aten.sub_.Tensor,): CortexMAddMulCheck,
    (torch.ops.aten.relu.default,): NsUnaryActivationCheck,
    (torch.ops.aten.relu_.default,): NsUnaryActivationCheck,
    (torch.ops.aten.hardtanh.default,): NsUnaryActivationCheck,
    (torch.ops.aten.hardtanh_.default,): NsUnaryActivationCheck,
    (torch.ops.aten.clamp.default,): NsUnaryActivationCheck,
    (torch.ops.aten.clamp_.default,): NsUnaryActivationCheck,
    (torch.ops.aten.leaky_relu.default,): NsLeakyReluCheck,
    (torch.ops.aten.leaky_relu_.default,): NsLeakyReluCheck,
    (torch.ops.aten.mean.dim,): NsMeanCheck,
}

NS_QUANTIZER_SUPPORT_DICT = CORTEX_M_QUANTIZER_SUPPORT_DICT | NS_EXTRA_PATTERNS


class NsCortexMQuantizer(CortexMQuantizer):
    """CortexMQuantizer with the NS support dict and NS annotation passes."""

    def __init__(self) -> None:
        conv_targets: set[OpOverload] = set()
        for key in CONV_OP_PATTERNS.keys() | CONV_TRANSPOSE_OP_PATTERNS.keys():
            conv_targets.update(key)

        support_dict_name = __name__ + ".NS_QUANTIZER_SUPPORT_DICT"
        pattern_matcher = PatternMatcher(
            cast(
                dict[tuple[OpOverload, ...], Optional[type[PatternCheck]]],
                NS_QUANTIZER_SUPPORT_DICT,
            ),
            support_dict_name=support_dict_name,
        )
        quantizers: List[Quantizer] = [
            PatternQuantizer(
                INT8_PER_CHANNEL_CONFIG,
                node_finder=NodeTargetNodeFinder(list(conv_targets)),
                pattern_matcher=pattern_matcher,
            ),
            PatternQuantizer(
                INT8_PER_TENSOR_CONFIG,
                node_finder=GlobalNodeFinder(),
                pattern_matcher=pattern_matcher,
            ),
            SharedQspecQuantizer(),
        ]
        # Skip CortexMQuantizer.__init__ (which builds the stock quantizer
        # list); initialize the ComposableQuantizer base directly.
        ComposableQuantizer.__init__(self, quantizers)

    def annotate(self, model):
        reporter = QuantizerReporter(self.quantizers)
        model = ComposableQuantizer.annotate(self, model)
        reporter.log_quantizer_report(model)
        return model

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        _strip_guards_fn(model)
        pass_manager = NsCortexMPassManager(None)
        return pass_manager.transform_for_annotation(model)
