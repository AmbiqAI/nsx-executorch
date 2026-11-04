# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# ExecuTorch edge graph -> heliaAOT AIR.
#
# The frontend reads the quantized edge program after the NS pass list up to,
# but not including, the stock ConvertToCortexMPass (see
# lower_for_frontend()). At that point every quantized operator still carries
# its float input/output scales and zero points (node.meta input_qparams /
# output_qparams, from FoldAndAnnotateQParamsPass), activations are fused into
# output clamps, most operators are already cortex_m:: / cortex_m_ns:: nodes,
# and weights/biases are int8/int32 constants. That is the level of AIR's
# TFLite-shaped, fused operators, so each node maps to one AIR operator (or to
# none: zero pads and same-layout reshapes are aliases).
#
# Layout: AIR shapes are the physical (dim-order) shapes of the ExecuTorch
# tensors, so a channels_last 4-D tensor is NHWC in AIR and the bytes are the
# same on both sides. Axes and permutations are remapped accordingly.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from executorch.backends.transforms.utils import get_param_tensor, is_param_node

try:
    from helia_aot.air import AirModel
    from helia_aot.air.enums import (
        AirActivationType,
        AirOpType,
        AirPaddingType,
        AirReluType,
        AirTensorKind,
    )
    from helia_aot.air.operator import AirOperator
    from helia_aot.air.options import (
        AirAddOptions,
        AirAveragePool2DOptions,
        AirBatchMatMulOptions,
        AirConv2DOptions,
        AirDepthwiseConv2DOptions,
        AirFullyConnectedOptions,
        AirHardSwishOptions,
        AirLeakyReluOptions,
        AirMaximumOptions,
        AirMaxPool2DOptions,
        AirMeanOptions,
        AirMinimumOptions,
        AirMulOptions,
        AirPadOptions,
        AirQuantizeOptions,
        AirReluOptions,
        AirReshapeOptions,
        AirSoftmaxOptions,
        AirSubOptions,
        AirTransposeConvOptions,
        AirTransposeOptions,
    )
    from helia_aot.air.quantization import AirQuantizationParameters
    from helia_aot.air.tensor import AirTensor
except ImportError as exc:  # pragma: no cover - exercised only without helia-aot
    raise ImportError(
        "The helia-aot backend needs heliaAOT: pip install 'helia-torch[helia-aot]' (helia-aot>=1.3)"
    ) from exc

CHANNELS_LAST = (0, 2, 3, 1)


class UnsupportedNodeError(NotImplementedError):
    """A node the frontend cannot express in AIR (the partitioner should not have claimed it)."""


@dataclass(frozen=True)
class QParams:
    scale: float
    zp: int


def target_name(node: torch.fx.Node) -> str:
    """'namespace.op' for an edge-dialect call target, e.g. 'aten.convolution', 'cortex_m.pad'."""
    name = getattr(node.target, "_name", None) or str(node.target)
    return name.split(".")[0].replace("::", ".")


def dim_order(val: torch.Tensor) -> tuple[int, ...]:
    try:
        return tuple(int(d) for d in val.dim_order())
    except Exception:  # noqa: BLE001 - fake tensors without dim_order support
        return tuple(range(val.dim()))


def physical_shape(val: torch.Tensor) -> tuple[int, ...]:
    order = dim_order(val)
    return tuple(int(val.shape[d]) for d in order)


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """A real CPU tensor. The partitioner sees a fake program (shapes, no data): checks only need the
    shape and dtype, so a fake parameter becomes zeros."""
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(t, FakeTensor):
        return torch.zeros(tuple(t.shape), dtype=t.dtype)
    return t.detach().cpu()


def output_dtype(node) -> Optional[torch.dtype]:
    """The dtype a node really produces. Before ConvertToCortexMPass an aten operator still carries its
    float meta value; its output quantization says what it becomes."""
    oq = (node.meta.get("output_qparams") or {}).get(0) if node.op == "call_function" else None
    if oq is not None and getattr(oq, "dtype", None) is not None:
        return oq.dtype
    val = node.meta.get("val")
    return val.dtype if isinstance(val, torch.Tensor) else None


# Operators whose output has their input's quantization (no qparams of their own before ConvertToCortexMPass).
_QPARAM_PRESERVING = {"cortex_m.pad", "cortex_m.transpose", "aten.view_copy"}


def _user_input_q(placeholder) -> Optional["QParams"]:
    """A delegate input's quantization: from the first consumer that records it, looking through
    quantization-preserving operators (a zero pad before a pool, a reshape, a transpose)."""
    frontier = [placeholder]
    seen = set()
    while frontier:
        src = frontier.pop(0)
        for user in src.users:
            if user.op != "call_function" or user.name in seen:
                continue
            seen.add(user.name)
            iq = user.meta.get("input_qparams") or {}
            for i, arg in enumerate(user.args):
                if arg is src and i in iq:
                    return _qp(iq[i])
            if target_name(user) in _QPARAM_PRESERVING and user.args and user.args[0] is src:
                frontier.append(user)
    return None


def _qp(q) -> QParams:
    scale = q.scale if not isinstance(q.scale, (list, tuple)) else q.scale[0]
    zp = q.zp if not isinstance(q.zp, (list, tuple)) else q.zp[0]
    return QParams(float(scale), int(zp))


def _floats(v) -> list[float]:
    return [float(x) for x in v] if isinstance(v, (list, tuple)) else [float(v)]


def _ints(v) -> list[int]:
    return [int(x) for x in v] if isinstance(v, (list, tuple)) else [int(v)]


def _window_padding_type(in_hw, out_hw, kernel, stride, dilation) -> AirPaddingType:
    """AIR derives window-op output shapes from SAME/VALID while the kernels use the explicit pads.
    Pick the type whose LiteRT output size is the PyTorch graph's."""

    def out(n, k, s, d, same):
        return (n + s - 1) // s if same else (n + s - ((k - 1) * d + 1)) // s

    for kind, same in ((AirPaddingType.SAME, True), (AirPaddingType.VALID, False)):
        if all(out(i, k, s, d, same) == o for i, o, k, s, d in zip(in_hw, out_hw, kernel, stride, dilation)):
            return kind
    raise UnsupportedNodeError(f"no SAME/VALID padding type gives output {tuple(out_hw)} from {tuple(in_hw)}")


def fused_activation(out_q, zp: int) -> AirActivationType:
    """ActivationFusionPass folds a ReLU into the output clamp: qmin == zero point."""
    qmin, qmax = int(out_q.qmin), int(out_q.qmax)
    if qmax != 127:
        raise UnsupportedNodeError(f"fused clamp max {qmax} (relu6/hardtanh) has no AIR fused activation")
    if qmin == zp and zp > -128:
        return AirActivationType.RELU
    if qmin == -128:
        return AirActivationType.NONE
    raise UnsupportedNodeError(f"fused clamp [{qmin}, {qmax}] with zero point {zp}")


class _Builder:
    def __init__(self, ep, name: str):
        self.ep = ep
        self.name = name
        self.tensors: dict[str, AirTensor] = {}
        self.operators: list[AirOperator] = []
        self.node_tensor: dict[str, str] = {}
        self.tensor_q: dict[str, QParams] = {}
        self.tensor_order: dict[str, tuple[int, ...]] = {}  # activation tensor id -> its dim order
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.user_inputs = [
            s.arg.name for s in ep.graph_signature.input_specs if s.kind.name == "USER_INPUT"
        ]
        self.placeholders = {n.name: n for n in ep.graph_module.graph.nodes if n.op == "placeholder"}

    # -- tensors ---------------------------------------------------------------------------------------------

    def _tensor(self, *, kind, shape=None, dtype=None, data=None, scale=None, zp=None, qdim=0, name=None) -> str:
        tid = str(len(self.tensors))
        quant = None
        if scale is not None:
            quant = AirQuantizationParameters(
                scale=_floats(scale), zero_point=_ints(zp), quantization_dimension=qdim
            )
        kw: dict[str, Any] = dict(id=tid, kind=kind, quantization=quant, local_name=name or f"tensor_{tid}")
        if data is not None:
            kw["data"] = np.ascontiguousarray(data)
        else:
            kw["shape"] = tuple(shape)
            kw["dtype"] = dtype
            kw["shape_signature"] = tuple(shape)
        self.tensors[tid] = AirTensor(**kw)
        if scale is not None and not isinstance(scale, (list, tuple)):
            self.tensor_q[tid] = QParams(float(scale), int(zp))
        return tid

    def _activation(self, node, q: QParams, order: Optional[tuple[int, ...]] = None) -> str:
        """The AIR tensor for `node`'s output, in `order` (default: the node's own dim order)."""
        val = node.meta["val"]
        if output_dtype(node) != torch.int8:
            raise UnsupportedNodeError(f"{node.name}: {output_dtype(node)} activation (only int8 is delegated)")
        order = dim_order(val) if order is None else tuple(order)
        tid = self._tensor(kind=AirTensorKind.SCRATCH, shape=tuple(int(val.shape[d]) for d in order),
                           dtype=np.int8, scale=q.scale, zp=q.zp, name=node.name)
        self.tensor_order[tid] = order
        self.node_tensor[node.name] = tid
        return tid

    def _relayout(self, tid: str, order: tuple[int, ...], name: str) -> str:
        """`tid` in dim order `order`: itself, or a TRANSPOSE of it (same quantization)."""
        current = self.tensor_order.get(tid)
        if current is None or tuple(current) == tuple(order):
            return tid
        src = self.tensors[tid]
        logical = [0] * len(current)
        for pos, d in enumerate(current):
            logical[d] = src.shape[pos]
        q = self.tensor_q[tid]
        out = self._tensor(kind=AirTensorKind.SCRATCH, shape=tuple(logical[d] for d in order), dtype=np.int8,
                           scale=q.scale, zp=q.zp, name=name)
        self.tensor_order[out] = tuple(order)
        self._op(AirOpType.TRANSPOSE, [tid], [out],
                 AirTransposeOptions(permutations=[list(current).index(d) for d in order]))
        return out

    def _input_order(self, node, in_t: str) -> tuple[int, ...]:
        return self.tensor_order.get(in_t) or dim_order(node.meta["val"])

    def _finish_layout(self, node, out_t: str, order) -> None:
        """After a layout-agnostic op computed `out_t` in `order`: map the node to its own dim order."""
        wanted = dim_order(node.meta["val"])
        if tuple(order) != wanted:
            self.node_tensor[node.name] = self._relayout(out_t, wanted, f"{node.name}_relayout")

    def _out_q(self, node, fallback_from: Optional[torch.fx.Node] = None) -> QParams:
        oq = node.meta.get("output_qparams") or {}
        if 0 in oq:
            return _qp(oq[0])
        if fallback_from is not None:
            return self.tensor_q[self.node_tensor[fallback_from.name]]
        raise UnsupportedNodeError(f"{node.name}: no output quantization")

    def _input(self, node, arg_index: int, operand) -> str:
        """AIR tensor for an operand: a produced activation, a user input, or a constant."""
        if not isinstance(operand, torch.fx.Node):
            raise UnsupportedNodeError(f"{node.name}: operand {arg_index} is not a tensor")
        if operand.name in self.node_tensor:
            return self.node_tensor[operand.name]
        if operand.op == "placeholder" and operand.name in self.user_inputs:
            iq = node.meta.get("input_qparams") or {}
            q = _qp(iq[arg_index]) if arg_index in iq else _user_input_q(operand)
            if q is None:
                raise UnsupportedNodeError(f"{node.name}: no input quantization for user input {operand.name}")
            return self._user_input(operand, q)
        if is_param_node(self.ep, operand):
            iq = node.meta.get("input_qparams") or {}
            data = _materialize(get_param_tensor(self.ep, operand))
            if data.dtype != torch.int8 or arg_index not in iq:
                raise UnsupportedNodeError(f"{node.name}: constant operand {operand.name} is not int8-quantized")
            q = _qp(iq[arg_index])
            tid = self._tensor(kind=AirTensorKind.CONSTANT, data=self._physical_data(data, operand), scale=q.scale,
                               zp=q.zp, name=operand.name)
            self.node_tensor[operand.name] = tid
            return tid
        raise UnsupportedNodeError(f"{node.name}: operand {operand.name} has no AIR tensor")

    def _physical_data(self, data: torch.Tensor, node) -> np.ndarray:
        val = node.meta.get("val")
        order = dim_order(val) if val is not None else tuple(range(data.dim()))
        return data.permute(order).contiguous().numpy()

    def _user_input(self, placeholder, q: QParams) -> str:
        val = placeholder.meta["val"]
        if val.dtype != torch.int8:
            raise UnsupportedNodeError(f"user input {placeholder.name} is {val.dtype}; delegate inputs are int8")
        tid = self._tensor(kind=AirTensorKind.SCRATCH, shape=physical_shape(val), dtype=np.int8,
                           scale=q.scale, zp=q.zp, name=placeholder.name)
        self.tensor_order[tid] = dim_order(val)
        self.node_tensor[placeholder.name] = tid
        return tid

    def _const(self, data: np.ndarray, *, scale=None, zp=None, qdim=0, name=None) -> str:
        return self._tensor(kind=AirTensorKind.CONSTANT, data=data, scale=scale, zp=zp, qdim=qdim, name=name)

    def _param(self, node) -> torch.Tensor:
        if node is None or not isinstance(node, torch.fx.Node) or not is_param_node(self.ep, node):
            raise UnsupportedNodeError(f"{node}: expected a constant parameter")
        return _materialize(get_param_tensor(self.ep, node))

    def _op(self, op_type, inputs, outputs, options, named=None):
        self.operators.append(AirOperator(id=str(len(self.operators)), op_type=op_type, input_ids=list(inputs),
                                          output_ids=list(outputs), named_tensors=named or {}, options=options))

    def _alias(self, node, source) -> None:
        self.node_tensor[node.name] = self.node_tensor[source.name] if source.name in self.node_tensor else \
            self._input(node, 0, source)

    def _require_channels_last(self, node, *vals):
        for val in vals:
            if val.dim() == 4 and dim_order(val) != CHANNELS_LAST:
                raise UnsupportedNodeError(f"{node.name}: needs channels_last tensors")

    # -- operators -------------------------------------------------------------------------------------------

    def convolution(self, node):
        x, w, b, stride, padding, dilation, transposed, out_pad, groups = node.args
        self._require_channels_last(node, x.meta["val"], node.meta["val"])
        iq = node.meta["input_qparams"]
        oq = _qp(node.meta["output_qparams"][0])
        weight = self._param(w)
        bias = self._param(b) if b is not None else None
        in_t = self._input(node, 0, x)
        out_t = self._activation(node, oq)
        act = fused_activation(node.meta["output_qparams"][0], oq.zp)
        wq = iq[1]
        named = {}
        if bias is not None:
            named["bias"] = self._const(bias.numpy().astype(np.int32), scale=iq[2].scale, zp=iq[2].zp, qdim=0,
                                        name=f"{node.name}_bias")
        in_hw = self.tensors[in_t].shape[1:3]
        out_hw = self.tensors[out_t].shape[1:3]
        if transposed:
            # PyTorch crops `padding` on both sides; CMSIS-NN crops `padding` at the top/left and
            # `padding + offset` at the bottom/right. output_padding would need a negative offset.
            if groups != 1 or any(int(p) for p in out_pad):
                raise UnsupportedNodeError(f"{node.name}: grouped or output_padding transposed convolution")
            in_ch, out_ch, kh, kw = weight.shape  # (in, out, H, W)
            named["weights"] = self._const(weight.permute(1, 2, 3, 0).numpy().astype(np.int8), scale=wq.scale,
                                           zp=wq.zp, qdim=0, name=f"{node.name}_weights")
            self._op(AirOpType.TRANSPOSE_CONV, [in_t], [out_t], AirTransposeConvOptions(
                padding_height=int(padding[0]), padding_width=int(padding[1]),
                padding_offsets_height=0, padding_offsets_width=0,
                stride_height=int(stride[0]), stride_width=int(stride[1]),
                dilation_height=int(dilation[0]), dilation_width=int(dilation[1]),
                activation=act, output_shape=list(self.tensors[out_t].shape)), named)
            return
        out_ch, in_per_group, kh, kw = weight.shape  # OIHW
        in_ch = in_per_group * groups
        if groups not in (1, in_ch):
            raise UnsupportedNodeError(f"{node.name}: grouped convolution (groups={groups})")
        depthwise = groups > 1
        common = dict(padding_height=int(padding[0]), padding_width=int(padding[1]),
                      stride_height=int(stride[0]), stride_width=int(stride[1]),
                      dilation_height=int(dilation[0]), dilation_width=int(dilation[1]), activation=act,
                      padding=_window_padding_type(in_hw, out_hw, (kh, kw), stride, dilation))
        if depthwise:
            named["weights"] = self._const(weight.permute(1, 2, 3, 0).numpy().astype(np.int8), scale=wq.scale,
                                           zp=wq.zp, qdim=3, name=f"{node.name}_weights")
            self._op(AirOpType.DEPTHWISE_CONV_2D, [in_t], [out_t],
                     AirDepthwiseConv2DOptions(depth_multiplier=out_ch // in_ch, **common), named)
        else:
            named["weights"] = self._const(weight.permute(0, 2, 3, 1).numpy().astype(np.int8), scale=wq.scale,
                                           zp=wq.zp, qdim=0, name=f"{node.name}_weights")
            self._op(AirOpType.CONV_2D, [in_t], [out_t], AirConv2DOptions(**common), named)

    def linear(self, node):
        x, w = node.args[:2]
        b = node.args[2] if len(node.args) > 2 else None
        iq = node.meta["input_qparams"]
        oq = _qp(node.meta["output_qparams"][0])
        in_t = self._input(node, 0, x)
        out_t = self._activation(node, oq)
        named = {"weights": self._const(self._param(w).numpy().astype(np.int8), scale=iq[1].scale, zp=iq[1].zp,
                                        qdim=0, name=f"{node.name}_weights")}
        if b is not None:
            named["bias"] = self._const(self._param(b).numpy().astype(np.int32), scale=iq[2].scale, zp=iq[2].zp,
                                        qdim=0, name=f"{node.name}_bias")
        self._op(AirOpType.FULLY_CONNECTED, [in_t], [out_t], AirFullyConnectedOptions(
            activation=fused_activation(node.meta["output_qparams"][0], oq.zp), keep_num_dims=
            len(node.meta["val"].shape) > 2), named)

    def bmm(self, node):
        a, b = node.args
        lhs, rhs = self._input(node, 0, a), self._input(node, 1, b)
        out_t = self._activation(node, self._out_q(node))
        self._op(AirOpType.BATCH_MATMUL, [lhs, rhs], [out_t], AirBatchMatMulOptions(adj_x=False, adj_y=False))

    def binary(self, node, op_type, options_cls, lhs_index, rhs_index, with_activation=True):
        lhs = self._input(node, 0, node.args[lhs_index])
        rhs = self._input(node, 1, node.args[rhs_index])
        order = self.tensor_order.get(lhs) or dim_order(node.meta["val"])
        if len(self.tensors[rhs].shape) == len(order):
            rhs = self._relayout(rhs, order, f"{node.name}_rhs_relayout")
        oq = self._out_q(node)
        options = options_cls(activation=fused_activation(node.meta["output_qparams"][0], oq.zp)) \
            if with_activation else options_cls()
        out_t = self._activation(node, oq, order=order)
        self._op(op_type, [lhs, rhs], [out_t], options)
        self._finish_layout(node, out_t, order)

    def relu(self, node):
        # cortex_m_ns.quantized_relu(x, in_zp, out_zp, mult, shift, act_min, act_max): the clamp bounds
        # in the output domain. Map them back to the AIR ReLU family.
        x = node.args[0]
        act_min, act_max = int(node.args[5]), int(node.args[6])
        in_t = self._input(node, 0, x)
        oq = self._out_q(node)
        order = self._input_order(node, in_t)
        out_t = self._activation(node, oq, order=order)
        # The clamp's real bounds; an unclamped top (127) means no upper bound.
        lo = (act_min - oq.zp) * oq.scale
        hi = (act_max - oq.zp) * oq.scale if act_max < 127 else np.inf
        tol = oq.scale

        def near(v, t):
            return (np.isinf(v) and np.isinf(t)) or abs(v - t) <= tol

        for kind, bounds in ((AirReluType.RELU, (0.0, np.inf)), (AirReluType.RELU6, (0.0, 6.0)),
                             (AirReluType.RELU_N1_TO_1, (-1.0, 1.0)), (AirReluType.RELU_0_TO_1, (0.0, 1.0))):
            if near(lo, bounds[0]) and near(hi, bounds[1]):
                self._op(AirOpType.RELU, [in_t], [out_t], AirReluOptions(relu_type=kind))
                self._finish_layout(node, out_t, order)
                return
        raise UnsupportedNodeError(f"{node.name}: clamp [{lo}, {hi}] is not an AIR ReLU variant")

    def leaky_relu(self, node):
        x, in_zp, out_zp, alpha_mult, alpha_shift, id_mult, id_shift = node.args
        in_t = self._input(node, 0, x)
        iq = self.tensor_q[in_t]
        oq = self._out_q(node)
        order = self._input_order(node, in_t)
        out_t = self._activation(node, oq, order=order)
        # helia-torch lowered alpha as quantize_multiplier_aot(in_scale * alpha / out_scale).
        alpha = (int(alpha_mult) / float(1 << 31)) * (2.0 ** int(alpha_shift)) * oq.scale / iq.scale
        self._op(AirOpType.LEAKY_RELU, [in_t], [out_t], AirLeakyReluOptions(alpha=alpha))
        self._finish_layout(node, out_t, order)

    def unary(self, node, op_type, options, layout_agnostic=False):
        in_t = self._input(node, 0, node.args[0])
        if layout_agnostic and in_t in self.tensor_order:
            order = self.tensor_order[in_t]
            out_t = self._activation(node, self._out_q(node), order=order)
            self._op(op_type, [in_t], [out_t], options)
            self._finish_layout(node, out_t, order)
            return
        out_t = self._activation(node, self._out_q(node))
        self._op(op_type, [in_t], [out_t], options)

    def mean(self, node):
        x, _in_zp, axis_mask, keepdim = node.args[:4]
        in_val = x.meta["val"]
        rank = in_val.dim()
        mask = list(axis_mask)[4 - rank:]  # right-aligned 4-element mask over the logical dims
        logical_axes = [i for i, m in enumerate(mask) if m]
        order = dim_order(in_val)
        axes = sorted(order.index(a) for a in logical_axes)
        if not keepdim and order != tuple(range(rank)):
            raise UnsupportedNodeError(f"{node.name}: keepdim=False mean over a non-contiguous tensor")
        self.unary(node, AirOpType.MEAN, AirMeanOptions(axis=axes, keep_dims=bool(keepdim)))

    def softmax(self, node):
        x, dim = node.args[:2]
        val = x.meta["val"]
        order = dim_order(val)
        if order.index(int(dim) % val.dim()) != val.dim() - 1:
            raise UnsupportedNodeError(f"{node.name}: softmax over a non-innermost axis")
        self.unary(node, AirOpType.SOFTMAX, AirSoftmaxOptions(beta=1.0))

    def pool(self, node, max_pool: bool):
        x = node.args[0]
        self._require_channels_last(node, x.meta["val"], node.meta["val"])
        kernel, stride, padding = node.args[1], node.args[2], node.args[3]
        if max_pool:
            dilation, ceil_mode = node.args[4], node.args[5]
            if any(int(d) != 1 for d in dilation) or ceil_mode:
                raise UnsupportedNodeError(f"{node.name}: dilated or ceil_mode max pool")
        in_t = self._input(node, 0, x)
        out_t = self._activation(node, self._out_q(node, fallback_from=x))
        in_hw, out_hw = self.tensors[in_t].shape[1:3], self.tensors[out_t].shape[1:3]
        common = dict(padding_height=int(padding[0]), padding_width=int(padding[1]),
                      stride_height=int(stride[0]), stride_width=int(stride[1]),
                      filter_height=int(kernel[0]), filter_width=int(kernel[1]),
                      activation=AirActivationType.NONE,
                      padding=_window_padding_type(in_hw, out_hw, kernel, stride, (1, 1)))
        if max_pool:
            self._op(AirOpType.MAX_POOL_2D, [in_t], [out_t], AirMaxPool2DOptions(**common))
        else:
            if any(int(p) for p in padding):
                raise UnsupportedNodeError(f"{node.name}: average pool padding (helia-torch emits a pad op)")
            self._op(AirOpType.AVERAGE_POOL_2D, [in_t], [out_t], AirAveragePool2DOptions(**common))

    def pad(self, node):
        x, pre, post, value = node.args[:4]
        if not any(int(p) for p in list(pre) + list(post)):
            self._alias(node, x)  # helia-torch emits zero pads before pooling: a copy
            return
        in_t = self._input(node, 0, x)
        out_t = self._activation(node, self._out_q(node, fallback_from=x))
        rank = len(self.tensors[in_t].shape)
        pre_p, post_p = [int(p) for p in pre][-rank:], [int(p) for p in post][-rank:]
        self._op(AirOpType.PAD, [in_t], [out_t],
                 AirPadOptions(pre_paddings=pre_p, post_paddings=post_p, constant_value=int(value)))

    def transpose(self, node):
        x, perm = node.args[:2]
        in_order, out_order = dim_order(x.meta["val"]), dim_order(node.meta["val"])
        physical = [in_order.index(int(perm[d])) for d in out_order]
        in_t = self._input(node, 0, x)
        out_t = self._activation(node, self._out_q(node, fallback_from=x))
        self._op(AirOpType.TRANSPOSE, [in_t], [out_t], AirTransposeOptions(permutations=physical))

    def reshape(self, node):
        x = node.args[0]
        in_val, out_val = x.meta["val"], node.meta["val"]
        in_order, out_order = dim_order(in_val), dim_order(out_val)

        def identity(order):
            return order == tuple(range(len(order)))

        # Same bytes on both sides only if both are contiguous, or the channels_last input has unit H, W.
        if not ((identity(in_order) and identity(out_order)) or
                (in_order == CHANNELS_LAST and int(in_val.shape[2]) == 1 and int(in_val.shape[3]) == 1
                 and identity(out_order))):
            raise UnsupportedNodeError(f"{node.name}: reshape that reorders bytes between dim orders")
        in_t = self._input(node, 0, x)
        out_t = self._activation(node, self._out_q(node, fallback_from=x))
        shape_t = self._const(np.array(physical_shape(out_val), dtype=np.int32), name=f"{node.name}_shape")
        self._op(AirOpType.RESHAPE, [in_t], [out_t], AirReshapeOptions(new_shape=[]), {"shape": shape_t})

    def quantize(self, node):
        src, scale, zp = node.args[:3]
        if isinstance(src, torch.fx.Node) and target_name(src) == "cortex_m.dequantize_per_tensor":
            # dequantize -> quantize between int8 tensors: a requantize.
            in_t = self.node_tensor[src.args[0].name]
            order = self._input_order(node, in_t)
            out_t = self._activation(node, QParams(float(scale), int(zp)), order=order)
            self._op(AirOpType.QUANTIZE, [in_t], [out_t], AirQuantizeOptions())
            self._finish_layout(node, out_t, order)
            return
        if isinstance(src, torch.fx.Node) and src.op == "placeholder" and src.name in self.user_inputs:
            # A float model input: the AIR model takes the quantized int8 tensor directly.
            val = node.meta["val"]
            tid = self._tensor(kind=AirTensorKind.SCRATCH, shape=physical_shape(val), dtype=np.int8,
                               scale=float(scale), zp=int(zp), name=src.name)
            self.node_tensor[node.name] = tid
            self.node_tensor[src.name] = tid
            return
        raise UnsupportedNodeError(f"{node.name}: quantize of a float tensor inside the graph")

    def dequantize(self, node):
        users = list(node.users)
        if users and all(target_name(u) == "cortex_m.quantize_per_tensor" for u in users):
            return  # consumed by quantize() as a requantize
        if all(u.op == "output" for u in users):
            self.node_tensor[node.name] = self.node_tensor[node.args[0].name]  # model output stays int8
            return
        raise UnsupportedNodeError(f"{node.name}: dequantize to float inside the graph")

    def handlers(self) -> dict[str, Callable]:
        return {
            "aten.convolution": self.convolution,
            "aten.linear": self.linear,
            "aten.bmm": self.bmm,
            "aten.view_copy": self.reshape,
            "cortex_m.quantized_add": lambda n: self.binary(n, AirOpType.ADD, AirAddOptions, 0, 4),
            "cortex_m_ns.quantized_sub": lambda n: self.binary(n, AirOpType.SUB, AirSubOptions, 0, 4),
            "cortex_m.quantized_mul": lambda n: self.binary(n, AirOpType.MUL, AirMulOptions, 0, 2),
            "cortex_m.maximum": lambda n: self.binary(n, AirOpType.MAXIMUM, AirMaximumOptions, 0, 1, False),
            "cortex_m.minimum": lambda n: self.binary(n, AirOpType.MINIMUM, AirMinimumOptions, 0, 1, False),
            "cortex_m_ns.quantized_relu": self.relu,
            "cortex_m_ns.quantized_leaky_relu": self.leaky_relu,
            "cortex_m_ns.quantized_hardswish": lambda n: self.unary(n, AirOpType.HARD_SWISH, AirHardSwishOptions(),
                                                                    layout_agnostic=True),
            "cortex_m_ns.quantized_mean": self.mean,
            "cortex_m.softmax": self.softmax,
            "cortex_m.quantized_max_pool2d": lambda n: self.pool(n, True),
            "cortex_m.quantized_avg_pool2d": lambda n: self.pool(n, False),
            "cortex_m.pad": self.pad,
            "cortex_m.transpose": self.transpose,
            "cortex_m.quantize_per_tensor": self.quantize,
            "cortex_m.dequantize_per_tensor": self.dequantize,
        }

    def build(self) -> AirModel:
        h = self.handlers()
        graph = self.ep.graph_module.graph
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            handler = h.get(target_name(node))
            if handler is None:
                raise UnsupportedNodeError(f"{node.name}: {target_name(node)} has no AIR mapping")
            handler(node)
        # Model inputs in the program's user-input order; outputs in output order.
        for name in self.user_inputs:
            if name not in self.node_tensor:
                raise UnsupportedNodeError(f"user input {name} is unused or not int8")
            self.inputs.append(self.node_tensor[name])
        out_node = next(n for n in graph.nodes if n.op == "output")
        for arg in out_node.args[0]:
            self.outputs.append(self.node_tensor[arg.name])
        return AirModel(name=self.name, description="Lowered from an ExecuTorch edge program by helia-torch",
                        tensors=self.tensors, operators=self.operators, input_ids=self.inputs,
                        output_ids=self.outputs)


def edge_to_air(ep, name: str = "model") -> AirModel:
    """Convert a quantized edge ExportedProgram, lowered with lower_for_frontend(), to an AirModel."""
    return _Builder(ep, name).build()


def lower_for_frontend(ep, target_config=None):
    """Run the NS pass list up to (not including) ConvertToCortexMPass: the frontend's input form."""
    from executorch.backends.cortex_m.passes.convert_to_cortex_m_pass import ConvertToCortexMPass

    from ..pass_manager_ns import NS_PASS_LIST, NsCortexMPassManager

    cut = NS_PASS_LIST.index(ConvertToCortexMPass)
    return NsCortexMPassManager(ep, passes=NS_PASS_LIST[:cut], target_config=target_config).transform()


# -- support checks (used by the partitioner) -------------------------------------------------------------------

# Quantize/dequantize are only delegated as an internal requantize pair (dequantize -> quantize); the
# float<->int8 conversions at a partition's edges stay ExecuTorch cortex_m kernels, so every delegate
# exchanges int8 tensors.
_REQUANT_Q = "cortex_m.quantize_per_tensor"
_REQUANT_DQ = "cortex_m.dequantize_per_tensor"


def _is_requant_pair_member(node) -> bool:
    t = target_name(node)
    if t == _REQUANT_Q:
        src = node.args[0]
        return isinstance(src, torch.fx.Node) and src.op == "call_function" and target_name(src) == _REQUANT_DQ
    if t == _REQUANT_DQ:
        users = list(node.users)
        return bool(users) and all(u.op == "call_function" and target_name(u) == _REQUANT_Q for u in users)
    return False


def unsupported_reason(ep, node) -> Optional[str]:
    """Why `node` cannot be delegated, or None when the frontend converts it.

    Runs the node's own frontend handler in isolation, with stand-in AIR tensors for its operands, so the
    partitioner and the frontend can never disagree about what is supported.
    """
    if node.op != "call_function":
        return "not an operator"
    t = target_name(node)
    if t in (_REQUANT_Q, _REQUANT_DQ):
        return None if _is_requant_pair_member(node) else "float<->int8 conversion stays in ExecuTorch"
    builder = _Builder(ep, "check")
    handler = builder.handlers().get(t)
    if handler is None:
        return f"{t} has no AIR mapping"
    if output_dtype(node) != torch.int8:
        return f"output is {output_dtype(node)}, not int8"
    iq = node.meta.get("input_qparams") or {}
    for i, arg in enumerate(node.args):
        if not isinstance(arg, torch.fx.Node) or (arg.op == "placeholder" and is_param_node(ep, arg)):
            continue
        aval = arg.meta.get("val")
        if not isinstance(aval, torch.Tensor):
            continue
        if output_dtype(arg) != torch.int8:
            return f"operand {arg.name} is {output_dtype(arg)}, not int8"
        oq = (arg.meta.get("output_qparams") or {}).get(0) if arg.op == "call_function" else None
        q = _qp(iq[i]) if i in iq else (_qp(oq) if oq is not None else QParams(1.0, 0))
        builder.node_tensor[arg.name] = builder._tensor(
            kind=AirTensorKind.SCRATCH, shape=physical_shape(aval), dtype=np.int8, scale=q.scale, zp=q.zp,
            name=arg.name)
    try:
        handler(node)
    except UnsupportedNodeError as exc:
        return str(exc)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def produces_air_operator(node) -> bool:
    """False for nodes the frontend turns into aliases (zero pads), which alone make an empty partition."""
    if target_name(node) == "cortex_m.pad":
        return any(int(p) for p in list(node.args[1]) + list(node.args[2]))
    return target_name(node) != _REQUANT_DQ


def retype_quantized_outputs(ep) -> int:
    """Make each quantized node's meta value carry the dtype it really produces.

    Before ConvertToCortexMPass an aten operator still holds its float (or int32 accumulator) meta value; the
    stock pass fixes that when it rewrites the node. Nodes delegated instead never get rewritten, and their
    meta values become the delegate's output tensors in the program -- so retype them before partitioning.
    Returns the number of nodes changed.
    """
    changed = 0
    for node in ep.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        val = node.meta.get("val")
        dtype = output_dtype(node)
        if isinstance(val, torch.Tensor) and dtype is not None and val.dtype != dtype:
            node.meta["val"] = val.to(dtype)
            changed += 1
    return changed
