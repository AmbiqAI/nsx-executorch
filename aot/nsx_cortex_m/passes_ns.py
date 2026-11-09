# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# Edge-dialect rewrite passes that lower quantized aten ops to `cortex_m_ns::`
# kernels. Every qualifier failure leaves the original aten op untouched so
# unsupported cases stay on the (correct but slow) portable fallback path —
# never a broken cortex_m_ns op.

import logging

import torch
from executorch.backends.cortex_m.passes.passes_utils import (
    extract_constant_scalar,
    is_channels_last,
    quantize_multiplier_aot,
    quantize_val,
    SHIFT_INT8,
)
from executorch.backends.cortex_m.passes.quantized_op_fusion_pass import (
    QuantizedOpFusionPass,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_manager import PassResult

from .operators_ns import (  # noqa: F401  (registers the cortex_m_ns library)
    IDENTITY_REQUANT_MULTIPLIER,
    IDENTITY_REQUANT_SHIFT,
)
from .layer_norm_ns import LayerNormNotSupported
from .layer_norm_ns import export_params as export_layer_norm_params

logger = logging.getLogger(__name__)

# Guard against int32 overflow of x * xr in the precise hardswish kernel:
# |x| <= 255 so relu_q6 must stay well below 2^31 / 255.
_HARDSWISH_MAX_RELU_Q6 = 1 << 22


def _get_qparams(meta):
    """Return (input_qparams, output_qparams) lists or None if missing."""
    input_qparams = meta.data.get("input_qparams", {})
    output_qparams = meta.data.get("output_qparams", {})
    if not input_qparams or not output_qparams:
        return None
    return input_qparams, output_qparams


def _requant_pair(input_scale: float, output_scale: float, input_zp: int, output_zp: int):
    """Requantization pair for input_scale/output_scale; identity sentinel
    when the quantization domains match exactly."""
    if input_scale == output_scale and input_zp == output_zp:
        return IDENTITY_REQUANT_MULTIPLIER, IDENTITY_REQUANT_SHIFT
    return quantize_multiplier_aot(input_scale / output_scale)


class NsStubCapturePass(ExportPass):
    """Swap aten ops that cannot be re-traced with int8 fake tensors for
    AOT-only cortex_m_ns stub ops.

    Must run BEFORE FoldAndAnnotateQParamsPass: once q/dq are folded, the
    graph feeds int8 fake tensors into these ops and torch rejects them
    during ExportPass re-tracing (leaky_relu refuses the float
    negative_slope cast; mean refuses integral inputs outright). The stubs'
    fake metas are dtype-agnostic. NsQuantizedOpFusionPass later fuses each
    stub into its quantized cortex_m_ns op or reverts it to the aten op.
    """

    _LEAKY_RELU_TARGETS = (
        exir_ops.edge.aten.leaky_relu.default,
        exir_ops.edge.aten.leaky_relu_.default,
    )

    def call_operator(self, op, args, kwargs, meta):
        if op in self._LEAKY_RELU_TARGETS:
            alpha = args[1] if len(args) > 1 else kwargs.get("negative_slope", 0.01)
            return super().call_operator(
                exir_ops.edge.cortex_m_ns.leaky_relu_stub.default,
                (args[0], alpha),
                {},
                meta,
            )
        if op == exir_ops.edge.aten.mean.dim:
            dims = args[1] if len(args) > 1 else kwargs.get("dim")
            keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            if isinstance(dims, int):
                dims = [dims]
            if dims and isinstance(keepdim, bool):
                return super().call_operator(
                    exir_ops.edge.cortex_m_ns.mean_stub.default,
                    (args[0], list(dims), keepdim),
                    {},
                    meta,
                )
        return super().call_operator(op, args, kwargs, meta)


class NsActivationRewritePass(ExportPass):
    """Rewrite standalone quantized relu/relu6/hardtanh/clamp to
    cortex_m_ns::quantized_relu.

    Must run after ActivationFusionPass (so producer-fusable activations are
    already consumed) and before the stock QuantizedClampActivationPass
    (whose output-domain clamp rewrite is only valid for shared input/output
    qparams; this pass handles differing qparams via requantization).
    """

    _TARGETS = {
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.relu_.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.hardtanh_.default,
        exir_ops.edge.aten.clamp.default,
    }

    def _get_float_bounds(self, op, args):
        if op in (exir_ops.edge.aten.relu.default, exir_ops.edge.aten.relu_.default):
            return (0.0, None)
        if op in (
            exir_ops.edge.aten.hardtanh.default,
            exir_ops.edge.aten.hardtanh_.default,
        ):
            min_val = extract_constant_scalar(args[1]) if len(args) > 1 else -1.0
            max_val = extract_constant_scalar(args[2]) if len(args) > 2 else 1.0
            if min_val is None or max_val is None:
                return None
            return (min_val, max_val)
        # clamp
        min_arg = args[1] if len(args) > 1 else None
        max_arg = args[2] if len(args) > 2 else None
        min_val = extract_constant_scalar(min_arg)
        max_val = extract_constant_scalar(max_arg)
        if (min_arg is not None and min_val is None) or (
            max_arg is not None and max_val is None
        ):
            return None  # non-scalar bounds
        return (min_val, max_val)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._TARGETS:
            return super().call_operator(op, args, kwargs, meta)

        qparams = _get_qparams(meta)
        if qparams is None:
            return super().call_operator(op, args, kwargs, meta)
        input_qparams, output_qparams = qparams

        input_qp = input_qparams[0]
        output_qp = output_qparams[0]
        if not isinstance(input_qp.scale, float) or not isinstance(
            output_qp.scale, float
        ):
            return super().call_operator(op, args, kwargs, meta)

        bounds = self._get_float_bounds(op, args)
        if bounds is None:
            logger.info("NS: %s bounds are not compile-time scalars; skipping", op)
            return super().call_operator(op, args, kwargs, meta)
        min_val, max_val = bounds

        qmin = output_qp.qmin
        qmax = output_qp.qmax
        act_min = (
            int(quantize_val(min_val, output_qp.scale, output_qp.zp, qmin, qmax))
            if min_val is not None
            else qmin
        )
        act_max = (
            int(quantize_val(max_val, output_qp.scale, output_qp.zp, qmin, qmax))
            if max_val is not None
            else qmax
        )

        multiplier, shift = _requant_pair(
            input_qp.scale, output_qp.scale, input_qp.zp, output_qp.zp
        )
        if not (-31 <= shift <= 31):
            return super().call_operator(op, args, kwargs, meta)

        new_args = (
            args[0],
            int(input_qp.zp),
            int(output_qp.zp),
            int(multiplier),
            int(shift),
            act_min,
            act_max,
        )
        return super().call_operator(
            exir_ops.edge.cortex_m_ns.quantized_relu.default, new_args, {}, meta
        )


class NsQuantizedOpFusionPass(QuantizedOpFusionPass):
    """Extends the stock fusion pass with cortex_m_ns:: lowerings for sub,
    hardswish, mean and leaky_relu."""

    def _get_sub_replacement(self, args, kwargs, meta):
        qparams = _get_qparams(meta)
        if qparams is None:
            return None
        input_qparams, output_qparams = qparams
        if len(input_qparams) < 2:
            return None
        alpha = kwargs.get("alpha", 1)
        if alpha != 1:
            return None

        scale1 = input_qparams[0].scale
        zero_point1 = input_qparams[0].zp
        scale2 = input_qparams[1].scale
        zero_point2 = input_qparams[1].zp
        output_scale = output_qparams[0].scale
        output_zero_point = output_qparams[0].zp

        # Same AOT math as the stock quantized_add lowering.
        max_scale_2x = 2 * max(scale1, scale2)
        input1_mult, input1_shift = quantize_multiplier_aot(scale1 / max_scale_2x)
        input2_mult, input2_shift = quantize_multiplier_aot(scale2 / max_scale_2x)
        output_mult, output_shift = quantize_multiplier_aot(
            max_scale_2x / (output_scale * (1 << SHIFT_INT8))
        )

        new_args = (
            args[0],
            zero_point1,
            input1_mult,
            input1_shift,
            args[1],
            zero_point2,
            input2_mult,
            input2_shift,
            output_zero_point,
            output_mult,
            output_shift,
            output_qparams[0].qmin,
            output_qparams[0].qmax,
        )
        return exir_ops.edge.cortex_m_ns.quantized_sub.default, new_args, {}

    def _get_hardswish_replacement(self, args, kwargs, meta):
        qparams = _get_qparams(meta)
        if qparams is None:
            return None
        input_qparams, output_qparams = qparams

        input_scale = float(input_qparams[0].scale)
        input_zp = int(input_qparams[0].zp)
        output_scale = float(output_qparams[0].scale)
        output_zp = int(output_qparams[0].zp)
        if input_scale <= 0.0 or output_scale <= 0.0:
            return None

        relu_q3 = int(round(3.0 / input_scale))
        relu_q6 = int(round(6.0 / input_scale))
        if relu_q6 < 1 or relu_q6 > _HARDSWISH_MAX_RELU_Q6:
            logger.info("NS: hardswish relu_q6=%d out of range; skipping", relu_q6)
            return None

        # prescale=0 matches the pinned ns-cmsis-nn unit tests and is
        # overflow-safe for all realistic int8 scales (guarded above).
        prescale = 0
        multiplier, shift = quantize_multiplier_aot(
            (input_scale * input_scale) / (6.0 * output_scale)
        )
        if not (-31 <= shift <= 31):
            return None

        new_args = (
            args[0],
            input_zp,
            output_zp,
            int(multiplier),
            int(shift),
            relu_q3,
            relu_q6,
            prescale,
        )
        return exir_ops.edge.cortex_m_ns.quantized_hardswish.default, new_args, {}

    def _get_mean_replacement(self, args, kwargs, meta):
        qparams = _get_qparams(meta)
        if qparams is None:
            return None
        input_qparams, output_qparams = qparams

        input_tensor = args[0].data
        rank = input_tensor.dim()
        if rank > 4 or is_channels_last(input_tensor):
            return None

        dims_arg = args[1] if len(args) > 1 else None
        if dims_arg is None:
            return None
        if isinstance(dims_arg, int):
            dims_arg = [dims_arg]
        try:
            dims = sorted({int(d) % rank for d in dims_arg})
        except (TypeError, ValueError):
            return None
        if not dims:
            return None

        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if not isinstance(keepdim, bool):
            return None

        axis_mask = [0, 0, 0, 0]
        count = 1
        offset = 4 - rank
        for d in dims:
            axis_mask[offset + d] = 1
            count *= int(input_tensor.shape[d])
        if count < 1:
            return None

        input_scale = float(input_qparams[0].scale)
        output_scale = float(output_qparams[0].scale)
        multiplier, shift = quantize_multiplier_aot(
            input_scale / (output_scale * count)
        )
        if not (-31 <= shift <= 31):
            return None

        new_args = (
            args[0],
            int(input_qparams[0].zp),
            axis_mask,
            keepdim,
            int(output_qparams[0].zp),
            int(multiplier),
            int(shift),
        )
        return exir_ops.edge.cortex_m_ns.quantized_mean.default, new_args, {}

    def _get_leaky_relu_replacement(self, args, kwargs, meta):
        qparams = _get_qparams(meta)
        if qparams is None:
            return None
        input_qparams, output_qparams = qparams

        alpha_arg = args[1] if len(args) > 1 else kwargs.get("negative_slope", 0.01)
        alpha = extract_constant_scalar(alpha_arg)
        if alpha is None or alpha <= 0.0:
            return None

        input_scale = float(input_qparams[0].scale)
        output_scale = float(output_qparams[0].scale)
        input_zp = int(input_qparams[0].zp)
        output_zp = int(output_qparams[0].zp)

        alpha_mult, alpha_shift = quantize_multiplier_aot(
            (input_scale * alpha) / output_scale
        )
        identity_mult, identity_shift = _requant_pair(
            input_scale, output_scale, input_zp, output_zp
        )
        if not (-31 <= alpha_shift <= 31) or not (-31 <= identity_shift <= 31):
            return None

        new_args = (
            args[0],
            input_zp,
            output_zp,
            int(alpha_mult),
            int(alpha_shift),
            int(identity_mult),
            int(identity_shift),
        )
        return exir_ops.edge.cortex_m_ns.quantized_leaky_relu.default, new_args, {}

    def call_operator(self, op, args, kwargs, meta):
        replacement = None
        if op == exir_ops.edge.aten.sub.Tensor:
            replacement = self._get_sub_replacement(args, kwargs, meta)
        elif op in (
            exir_ops.edge.aten.hardswish.default,
            exir_ops.edge.aten.hardswish_.default,
        ):
            replacement = self._get_hardswish_replacement(args, kwargs, meta)
        elif op == exir_ops.edge.aten.mean.dim:
            replacement = self._get_mean_replacement(args, kwargs, meta)
        elif op == exir_ops.edge.cortex_m_ns.mean_stub.default:
            replacement = self._get_mean_replacement(args, kwargs, meta)
            if replacement is None:
                # Revert the AOT-only stub to the original aten op. Safe:
                # without folded qparams the input is still float here.
                return ExportPass.call_operator(
                    self,
                    exir_ops.edge.aten.mean.dim,
                    args,
                    kwargs,
                    meta,
                )
        elif op == exir_ops.edge.cortex_m_ns.leaky_relu_stub.default:
            replacement = self._get_leaky_relu_replacement(args, kwargs, meta)
            if replacement is None:
                # Revert the AOT-only stub to the original aten op. Safe:
                # without folded qparams the input is still float here.
                return ExportPass.call_operator(
                    self,
                    exir_ops.edge.aten.leaky_relu.default,
                    args,
                    kwargs,
                    meta,
                )
        else:
            return super().call_operator(op, args, kwargs, meta)

        if replacement is None:
            # Qualifier failed: keep the original aten op (portable fallback),
            # preserving kwargs (e.g. keepdim) that the stock pass would drop.
            return ExportPass.call_operator(self, op, args, kwargs, meta)

        new_op, new_args, new_kwargs = replacement
        return ExportPass.call_operator(self, new_op, new_args, new_kwargs, meta)


class _NotConstant(Exception):
    """A layer-norm parameter that is not a graph constant."""


class NsLayerNormPass(XNNPACKPass):
    """Lower layer norm over the last dimension to ns-cmsis-nn's arm_layer_norm kernels.

    Runs before the stock ConvertToCortexMPass (so the heliaAOT frontend, which partitions at that point, sees the
    result). Two forms:

    - Quantized: the stock flow leaves a layer norm between quantized neighbours as
      cortex_m::dequantize -> aten.native_layer_norm -> getitem(0) -> cortex_m::quantize. That chain becomes one
      cortex_m_ns::quantized_layer_norm on the int8 (or int16) tensor, with gamma and beta folded with the input
      and output quantization into int32 constants (layer_norm_ns.export_params).
    - Float: any other float32 aten.native_layer_norm becomes cortex_m_ns::layer_norm (arm_layer_norm_f32)
      instead of the portable kernel.

    Only a layer norm over the last dimension of a contiguous tensor, whose mean and rstd outputs are unused, is
    lowered; one whose parameters the integer kernel cannot represent stays in float.
    """

    def _constant(self, node):
        if node is None:
            return None
        if not isinstance(node, torch.fx.Node) or not is_param_node(self.exported_program, node):
            raise _NotConstant
        return get_param_tensor(self.exported_program, node)

    @staticmethod
    def _only_output(ln: torch.fx.Node):
        """The getitem(ln, 0) node, if it is the only used output of the native_layer_norm."""
        out = None
        for user in ln.users:
            if user.op != "call_function" or "getitem" not in str(user.target):
                return None
            if user.args[1] == 0:
                out = user
            elif user.users:
                return None  # mean or rstd is used
        return out

    @staticmethod
    def _last_dim_contiguous(val: torch.Tensor, normalized_shape) -> bool:
        return (
            isinstance(val, torch.Tensor)
            and val.dim() >= 1
            and list(normalized_shape) == [int(val.shape[-1])]
            and val.is_contiguous()
        )

    def _lower_quantized(self, graph_module, ln, out) -> bool:
        dq = ln.args[0]
        if not (isinstance(dq, torch.fx.Node) and dq.target == exir_ops.edge.cortex_m.dequantize_per_tensor.default):
            return False
        users = list(out.users)
        if len(users) != 1 or users[0].target != exir_ops.edge.cortex_m.quantize_per_tensor.default:
            return False
        q = users[0]
        x_q, in_scale, _in_zp, _, _, in_dtype = dq.args[:6]
        _, out_scale, out_zp, _, _, out_dtype = q.args[:6]
        if in_dtype != out_dtype or in_dtype not in (torch.int8, torch.int16):
            return False
        if not self._last_dim_contiguous(x_q.meta.get("val"), ln.args[1]):
            return False
        try:
            gamma = self._constant(ln.args[2] if len(ln.args) > 2 else None)
            beta = self._constant(ln.args[3] if len(ln.args) > 3 else None)
        except _NotConstant:
            logger.debug("%s: gamma or beta is not a constant; keeping the float layer norm", ln.name)
            return False
        eps = float(ln.args[4]) if len(ln.args) > 4 else 1e-5
        try:
            p = export_layer_norm_params(gamma, beta, eps, float(in_scale), float(out_scale), int(out_zp),
                                         int(x_q.meta["val"].shape[-1]), in_dtype)
        except LayerNormNotSupported as exc:
            logger.info("%s: stays in float (%s)", ln.name, exc)
            return False

        # Constant placeholders must precede the user inputs: put them at the top of the graph.
        with graph_module.graph.inserting_before(next(iter(graph_module.graph.nodes))):
            gamma_node = create_constant_placeholder(
                self.exported_program, graph_module.graph, ln.name + "_ns_gamma", InputKind.PARAMETER, p.gamma)
            bias_node = create_constant_placeholder(
                self.exported_program, graph_module.graph, ln.name + "_ns_bias", InputKind.PARAMETER, p.bias)
        with graph_module.graph.inserting_before(q):
            ns_node = graph_module.graph.create_node(
                "call_function",
                target=exir_ops.edge.cortex_m_ns.quantized_layer_norm.default,
                args=(x_q, gamma_node, bias_node, p.epsilon, p.output_shift, p.activation_min, p.activation_max),
                kwargs={},
            )
        ns_node.meta = dict(q.meta)
        # The quantize node's folded qparams describe its float input; the new node's int8/int16 value is the truth.
        ns_node.meta.pop("input_qparams", None)
        ns_node.meta.pop("output_qparams", None)
        # The float parameters and quantization, for frontends that lower the operator again from its float
        # definition (the heliaAOT frontend emits AIR LAYER_NORM from them).
        ns_node.meta["ns_layer_norm"] = {
            "gamma": None if gamma is None else gamma.detach().to(torch.float32).clone(),
            "beta": None if beta is None else beta.detach().to(torch.float32).clone(),
            "eps": eps,
            "input_qparams": (float(in_scale), int(_in_zp)),
            "output_qparams": (float(out_scale), int(out_zp)),
        }
        q.replace_all_uses_with(ns_node)
        for node in (q, out):
            graph_module.graph.erase_node(node)
        return True

    def _lower_float(self, graph_module, ln, out) -> bool:
        x = ln.args[0]
        val = x.meta.get("val") if isinstance(x, torch.fx.Node) else None
        if not self._last_dim_contiguous(val, ln.args[1]) or val.dtype != torch.float32:
            return False
        weight = ln.args[2] if len(ln.args) > 2 else None
        bias = ln.args[3] if len(ln.args) > 3 else None
        eps = float(ln.args[4]) if len(ln.args) > 4 else 1e-5
        with graph_module.graph.inserting_before(out):
            ns_node = graph_module.graph.create_node(
                "call_function",
                target=exir_ops.edge.cortex_m_ns.layer_norm.default,
                args=(x, weight, bias, eps),
                kwargs={},
            )
        ns_node.meta = out.meta
        out.replace_all_uses_with(ns_node)
        graph_module.graph.erase_node(out)
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for ln in list(graph_module.graph.nodes):
            if ln.op != "call_function" or ln.target != exir_ops.edge.aten.native_layer_norm.default:
                continue
            out = self._only_output(ln)
            if out is None:
                continue
            if self._lower_quantized(graph_module, ln, out) or self._lower_float(graph_module, ln, out):
                modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
