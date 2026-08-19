# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# AOT support for the Ambiq ns-cmsis-nn additional kernels (`cortex_m_ns::`).
#
# Importing this package registers the cortex_m_ns torch.library operators.

from . import operators_ns  # noqa: F401  (registers the library)
from .export import export, ExportResult
from .pass_manager_ns import NS_ANNOTATION_PASS_LIST, NS_PASS_LIST, NsCortexMPassManager
from .passes_ns import NsActivationRewritePass, NsQuantizedOpFusionPass
from .quantizer_ns import NS_QUANTIZER_SUPPORT_DICT, NsCortexMQuantizer

__all__ = [
    "export",
    "ExportResult",
    "NS_ANNOTATION_PASS_LIST",
    "NS_PASS_LIST",
    "NS_QUANTIZER_SUPPORT_DICT",
    "NsActivationRewritePass",
    "NsCortexMPassManager",
    "NsCortexMQuantizer",
    "NsQuantizedOpFusionPass",
]
