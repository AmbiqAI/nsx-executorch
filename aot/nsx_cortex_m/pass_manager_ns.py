# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# NS profile pass lists and pass-manager factory. Composes the stock
# CortexMPassManager via its public `passes=` override — the pinned
# ExecuTorch tree is never modified.

from typing import Optional

from executorch.backends.arm._passes import (
    DeduplicateGetAttrPass,
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.cortex_m.passes.activation_fusion_pass import (
    ActivationFusionPass,
)
from executorch.backends.cortex_m.passes.convert_to_cortex_m_pass import (
    ConvertToCortexMPass,
)
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
    CortexMPassManager,
)
from executorch.backends.cortex_m.passes.decompose_mean_pass import DecomposeMeanPass
from executorch.backends.cortex_m.passes.quantized_clamp_activation_pass import (
    QuantizedClampActivationPass,
)
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
    ReplaceQuantNodesPass,
)
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from torch.export import ExportedProgram

from .passes_ns import (
    NsActivationRewritePass,
    NsQuantizedOpFusionPass,
    NsStubCapturePass,
)

# Differences from the stock CortexMPassManager.pass_list:
# - NsActivationRewritePass inserted after ActivationFusionPass so standalone
#   relu/relu6/hardtanh/clamp lower to cortex_m_ns::quantized_relu (with
#   requantization support) before the stock output-domain clamp rewrite.
# - DecomposeHardswishPass omitted: hardswish stays intact and lowers to
#   cortex_m_ns::quantized_hardswish in the fusion pass.
# - QuantizedOpFusionPass replaced by NsQuantizedOpFusionPass (adds sub,
#   hardswish, mean and leaky_relu lowerings on top of the stock cases).
NS_PASS_LIST = [
    RemoveGetItemPass,
    NsStubCapturePass,
    FoldAndAnnotateQParamsPass,
    ReplaceScalarWithTensorArgPass,
    ReplaceQuantNodesPass,
    ActivationFusionPass,
    NsActivationRewritePass,
    QuantizedClampActivationPass,
    NsQuantizedOpFusionPass,
    ConvertToCortexMPass,
]

# Differences from the stock pass_list_transform_for_annotation:
# - ClampHardswishPass omitted: the precise ns hardswish kernel handles the
#   full input range, so no observer-narrowing clamp is inserted.
# - DecomposeMeanPass kept: it decomposes adaptive_avg_pool2d (not aten.mean)
#   and removing it would regress adaptive pooling support.
NS_ANNOTATION_PASS_LIST = [
    ScalarsToAttributePass,
    ReplaceScalarWithTensorArgPass,
    DecomposeMeanPass,
    DeduplicateGetAttrPass,
]


class NsCortexMPassManager(CortexMPassManager):
    """Stock CortexMPassManager configured with the NS profile pass lists."""

    pass_list = NS_PASS_LIST
    pass_list_transform_for_annotation = NS_ANNOTATION_PASS_LIST

    def __init__(
        self,
        exported_program: ExportedProgram | None,
        passes=None,
        target_config: Optional[CortexMTargetConfig] = None,
    ) -> None:
        super().__init__(
            exported_program,
            passes=passes if passes is not None else NS_PASS_LIST,
            target_config=target_config,
        )
