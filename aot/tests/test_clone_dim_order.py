# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Regression coverage for nsx-executorch#11: dim_order_ops::_clone_dim_order
# has no Cortex-M lowering and always falls back to the portable kernel.
# NsCloneDimOrderRewritePass (aot/nsx_cortex_m/passes_ns.py) lowers the
# identity-dim_order case to cortex_m::transpose; see that pass's docstring
# for why a genuinely differing dim_order can't be lowered safely yet.

import torch
import torch.nn as nn

from nsx_cortex_m import export

_CLONE_DIM_ORDER = "dim_order_ops::_clone_dim_order"
_TRANSPOSE = "cortex_m::transpose"


class DepthwiseSeparableConv(nn.Module):
    """Mirrors executorch/examples/models/mlperf_tiny/ds_cnn.py's block
    (BatchNorm dropped: unrelated to this pass, and unfused BN keeps the
    CMSIS-NN conv pattern matcher from firing in a plain hand-built model,
    which would obscure the clone_dim_order behavior under test)."""

    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x


class DSCNNKWS(nn.Module):
    """The MLCommons Tiny Keyword Spotting DS-CNN shape (executorch/
    examples/models/mlperf_tiny/ds_cnn.py, BatchNorm dropped — see
    DepthwiseSeparableConv above). Exported with int8_io=True (this exact
    combination — full stem+blocks+pool+linear shape, int8 method
    boundary) reliably produces 2 dim_order_ops::_clone_dim_order nodes
    pre-fix, matching nsx-executorch#11's "2 invocations per inference on
    its int8 input path"; drop int8_io or truncate the model and they
    don't appear, so neither detail here is incidental."""

    def __init__(self, num_classes=12):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), padding=(5, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            DepthwiseSeparableConv(64),
            DepthwiseSeparableConv(64),
            DepthwiseSeparableConv(64),
            DepthwiseSeparableConv(64),
            nn.Dropout(p=0.4),
        )
        self.pool = nn.AvgPool2d(kernel_size=(24, 5))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _kws_example():
    torch.manual_seed(20260826)
    model = DSCNNKWS().eval()
    example = (torch.rand(1, 1, 49, 10) * 2 - 1,)
    return model, example


def test_clone_dim_order_lowers_to_transpose():
    model, example = _kws_example()

    result = export(model, example, kernel_provider="ns", int8_io=True)

    assert _CLONE_DIM_ORDER not in result.edge_ops, result.edge_ops
    assert result.edge_ops.get(_TRANSPOSE) == 2, result.edge_ops
    assert not result.portable_fallback_ops


def test_clone_dim_order_arm_provider_unaffected():
    """The arm kernel provider must stay byte-for-byte the stock flow:
    NsCloneDimOrderRewritePass is only ever wired into NS_PASS_LIST, so the
    clone nodes this model produces must survive untouched (portable)."""
    model, example = _kws_example()

    result = export(model, example, kernel_provider="arm", int8_io=True)

    assert _TRANSPOSE not in result.edge_ops, result.edge_ops
    assert result.edge_ops.get(_CLONE_DIM_ORDER) == 2, result.edge_ops
    ns_ops = [op for op in result.edge_ops if "cortex_m_ns" in op]
    assert not ns_ops, result.edge_ops
