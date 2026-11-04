# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Conv and depthwise conv lower to cortex_m_ns::*_ws with the ns-cmsis-nn
# weight sums precomputed at export time (NsConvWeightSumPass).

import pytest
import torch
from ns_tester import McuTestCase, NsCortexMTester, ramp_tensor

from nsx_cortex_m.export import export
from nsx_cortex_m.manifest import build_manifest
from nsx_cortex_m.weight_sums_ns import (
    conv_weight_sums,
    depthwise_weight_sums,
    WEIGHT_SUM_FORMAT,
)

_Q = "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
_DQ = "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
_DQ_CH = "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default"
_CONV = "executorch_exir_dialects_edge__ops_aten_convolution_default"
_CM_Q = "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default"
_CM_DQ = "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default"
_CM_CONV = "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default"
_CM_DW = "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_default"
_NS_CONV = "executorch_exir_dialects_edge__ops_cortex_m_ns_quantized_conv2d_ws_default"
_NS_DW = "executorch_exir_dialects_edge__ops_cortex_m_ns_quantized_depthwise_conv2d_ws_default"


# ---------------------------------------------------------------------------
# Weight-sum math against a direct transcription of the C reference loops.
# ---------------------------------------------------------------------------


def _wrap(value: int) -> int:
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def _conv_reference(weight_ohwi, bias, input_offset):
    out = []
    for o in range(weight_ohwi.shape[0]):
        acc = int(bias[o]) if bias is not None else 0
        acc = _wrap(acc + input_offset * int(weight_ohwi[o].to(torch.int64).sum()))
        out.append(acc)
    return torch.tensor(out, dtype=torch.int32)


def _depthwise_reference(weight_ihwo, bias, input_offset):
    channels = weight_ihwo.shape[3]
    out = []
    for c in range(channels):
        acc = int(bias[c]) if bias is not None else 0
        tap_sum = 0
        for h in range(weight_ihwo.shape[1]):
            for w in range(weight_ihwo.shape[2]):
                tap_sum += int(weight_ihwo[0, h, w, c])
        out.append(_wrap(acc + _wrap(tap_sum * input_offset)))
    return torch.tensor(out, dtype=torch.int32)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("with_bias", [True, False])
def test_conv_weight_sums_match_reference(seed, with_bias):
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randint(-128, 128, (16, 3, 3, 8), dtype=torch.int8, generator=gen)
    bias = torch.randint(-(1 << 20), 1 << 20, (16,), dtype=torch.int32, generator=gen) if with_bias else None
    offset = int(torch.randint(-127, 129, (1,), generator=gen))
    assert torch.equal(conv_weight_sums(weight, bias, offset), _conv_reference(weight, bias, offset))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("ch_mult", [1, 4])
def test_depthwise_weight_sums_match_reference(seed, ch_mult):
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randint(-128, 128, (1, 3, 3, 8 * ch_mult), dtype=torch.int8, generator=gen)
    bias = torch.randint(-(1 << 20), 1 << 20, (8 * ch_mult,), dtype=torch.int32, generator=gen)
    offset = int(torch.randint(-127, 129, (1,), generator=gen))
    assert torch.equal(
        depthwise_weight_sums(weight, bias, offset), _depthwise_reference(weight, bias, offset)
    )


def test_weight_sums_wrap_like_int32():
    # A bias near INT32_MAX plus a positive term must wrap, as the C kernels do.
    weight = torch.full((1, 1, 1, 64), 127, dtype=torch.int8)
    bias = torch.tensor([2**31 - 10], dtype=torch.int32)
    got = conv_weight_sums(weight, bias, 128)
    assert int(got[0]) == _wrap(2**31 - 10 + 128 * 127 * 64)
    assert int(got[0]) < 0


def test_weight_sums_reject_bad_layouts():
    with pytest.raises(ValueError):
        depthwise_weight_sums(torch.zeros(2, 3, 3, 8, dtype=torch.int8), None, 1)
    with pytest.raises(ValueError):
        conv_weight_sums(torch.zeros(8, 3, 3, 8, dtype=torch.float32), None, 1)
    with pytest.raises(ValueError):
        conv_weight_sums(torch.zeros(8, 1, 1, 8, dtype=torch.int8), torch.zeros(7, dtype=torch.int32), 1)


# ---------------------------------------------------------------------------
# Lowering: graph rewrite and host numerics. The _ws reference impls re-derive
# the sums from the lowered weights and fail on any mismatch, so these also
# check the constants the pass stored.
# ---------------------------------------------------------------------------


class Conv2D(torch.nn.Module):
    ops_before_transforms = {_CONV: 1, _Q: 2, _DQ: 2, _DQ_CH: 2}
    ops_after_transforms = {_NS_CONV: 1, _CM_Q: 1, _CM_DQ: 1}

    def __init__(self, cin, cout, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv2D(torch.nn.Module):
    ops_before_transforms = {_CONV: 1, _Q: 2, _DQ: 2, _DQ_CH: 2}
    ops_after_transforms = {_NS_DW: 1, _CM_Q: 1, _CM_DQ: 1}

    def __init__(self, channels, k, stride=1, padding=1, ch_mult=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            channels, channels * ch_mult, k, stride=stride, padding=padding, groups=channels
        )

    def forward(self, x):
        return self.conv(x)


class NoBiasConv2D(Conv2D):
    ops_before_transforms = {_CONV: 1, _Q: 2, _DQ: 2, _DQ_CH: 1}

    def __init__(self):
        super().__init__(8, 16, 3, padding=1, bias=False)


class DsCnnBlock(torch.nn.Module):
    """KWS DS-CNN shape: 1-channel stem (lowered as depthwise, ch_mult 64),
    then depthwise 3x3 + pointwise 1x1 with ReLU."""

    ops_before_transforms = {_CONV: 3}
    ops_after_transforms = {_NS_DW: 2, _NS_CONV: 1}

    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(1, 64, (10, 4), stride=2, padding=(5, 1))
        self.dw = torch.nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pw = torch.nn.Conv2d(64, 64, 1)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = torch.relu(self.dw(x))
        return torch.relu(self.pw(x))


def _nhwc(*shape):
    return ramp_tensor(-1, 1, shape).to(memory_format=torch.channels_last)


test_cases = {
    "conv3x3": McuTestCase(model=Conv2D(8, 16, 3, padding=1), example_inputs=(_nhwc(1, 8, 9, 7),)),
    "conv1x1": McuTestCase(model=Conv2D(16, 32, 1), example_inputs=(_nhwc(1, 16, 5, 5),)),
    "conv_stride2": McuTestCase(model=Conv2D(4, 8, 3, stride=2, padding=1), example_inputs=(_nhwc(1, 4, 10, 10),)),
    "conv_no_bias": McuTestCase(model=NoBiasConv2D(), example_inputs=(_nhwc(1, 8, 6, 6),)),
    "dw3x3_c64": McuTestCase(model=DepthwiseConv2D(64, 3), example_inputs=(_nhwc(1, 64, 25, 5),)),
    "dw3x3_s2": McuTestCase(model=DepthwiseConv2D(32, 3, stride=2), example_inputs=(_nhwc(1, 32, 12, 12),)),
    "dw5x5": McuTestCase(model=DepthwiseConv2D(16, 5, padding=2), example_inputs=(_nhwc(1, 16, 8, 8),)),
    "dw_ch_mult": McuTestCase(model=DepthwiseConv2D(8, 3, ch_mult=2), example_inputs=(_nhwc(1, 8, 6, 6),)),
}


@pytest.mark.parametrize("name", list(test_cases))
def test_dialect_conv_weight_sum(name):
    case = test_cases[name]
    tester = NsCortexMTester(case.model, case.example_inputs)
    tester.test_dialect(case.model.ops_before_transforms, case.model.ops_after_transforms, qtol=1)
    tester.check_not([_CM_CONV, _CM_DW])


def test_dialect_ds_cnn_block():
    model = DsCnnBlock()
    tester = NsCortexMTester(model, (_nhwc(1, 1, 49, 10),))
    tester.test_dialect(model.ops_before_transforms, model.ops_after_transforms, qtol=1)
    tester.check_not([_CM_CONV, _CM_DW])


# ---------------------------------------------------------------------------
# End to end: serialization and the sidecar.
# ---------------------------------------------------------------------------


def test_export_records_weight_sum_format():
    model = DsCnnBlock().eval()
    inputs = (_nhwc(1, 1, 49, 10),)
    result = export(model, inputs, kernel_provider="ns")
    manifest = build_manifest(result, result.executorch_program.buffer, "ns")
    assert manifest["requires_ns_ops"] is True
    assert manifest["weight_sum_format"] == WEIGHT_SUM_FORMAT
    ns_ops = manifest["operators"]["cortex_m_ns"]
    assert "cortex_m_ns::quantized_conv2d_ws.out" in ns_ops
    assert "cortex_m_ns::quantized_depthwise_conv2d_ws.out" in ns_ops
    assert not any("quantized_conv2d" in op or "depthwise" in op for op in manifest["operators"]["cortex_m"])


def test_arm_provider_keeps_stock_conv():
    model = DsCnnBlock().eval()
    inputs = (_nhwc(1, 1, 49, 10),)
    result = export(model, inputs, kernel_provider="arm")
    manifest = build_manifest(result, result.executorch_program.buffer, "arm")
    assert manifest["weight_sum_format"] is None
    assert manifest["operators"]["cortex_m_ns"] == []
    assert any(op.startswith("cortex_m::quantized_conv2d") for op in manifest["operators"]["cortex_m"])
