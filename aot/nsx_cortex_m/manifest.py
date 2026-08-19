# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# PTE sidecar manifests.
#
# A compiled model is only half of what a target build needs: the firmware
# must also register the exact kernel set (portable select-ops list, NS ops
# flag, provider), size three static arenas, and pre-allocate I/O buffers.
# All of those facts are known precisely once — at export time — so
# ExportResult.write_pte() records them in a `<model>.pte.json` sidecar that
# travels with the PTE. Consumers (helia-profiler, NSX app builds) verify
# `pte_sha256` and self-configure instead of asking a human to transcribe
# numbers whose failure mode is an on-device Method::load error.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "nsx-executorch.pte-manifest/1"

# executorch_flatbuffer ScalarType element sizes for I/O accounting.
_SCALAR_BYTES = {
    "BYTE": 1,
    "CHAR": 1,
    "BOOL": 1,
    "QINT8": 1,
    "QUINT8": 1,
    "SHORT": 2,
    "HALF": 2,
    "BFLOAT16": 2,
    "INT": 4,
    "FLOAT": 4,
    "QINT32": 4,
    "LONG": 8,
    "DOUBLE": 8,
}


def _operator_name(operator: Any) -> str:
    return operator.name + (f".{operator.overload}" if operator.overload else "")


def _tensor_facts(value: Any) -> dict[str, Any]:
    tensor = value.val
    dtype = tensor.scalar_type.name
    element_bytes = _SCALAR_BYTES.get(dtype)
    count = 1
    for dim in tensor.sizes:
        count *= dim
    return {
        "shape": list(tensor.sizes),
        "dtype": dtype,
        "size_bytes": count * element_bytes if element_bytes else None,
    }


def classify_operator(name: str) -> str:
    """Kernel class for one serialized operator name."""
    if name.startswith("cortex_m_ns::"):
        return "ns"
    if name.startswith("cortex_m::"):
        return "stock"
    return "portable"


def build_manifest(result: Any, pte_bytes: bytes, kernel_provider: str) -> dict[str, Any]:
    """Build the sidecar dict for an ExportResult and its serialized bytes."""
    program = result.executorch_program._emitter_output.program
    plan = program.execution_plan[0]
    operators = sorted({_operator_name(op) for op in plan.operators})
    by_class: dict[str, list[str]] = {"stock": [], "ns": [], "portable": []}
    for name in operators:
        by_class[classify_operator(name)].append(name)

    buffer_sizes = list(plan.non_const_buffer_sizes)
    planned = int(buffer_sizes[1]) if len(buffer_sizes) == 2 else None

    return {
        "schema": SCHEMA,
        "pte_sha256": hashlib.sha256(pte_bytes).hexdigest(),
        "pte_byte_size": len(pte_bytes),
        "kernel_provider": kernel_provider,
        "requires_ns_ops": bool(by_class["ns"]),
        "portable_select_ops_list": ",".join(by_class["portable"]),
        "planned_arena_size": planned,
        "inputs": [_tensor_facts(plan.values[index]) for index in plan.inputs],
        "outputs": [_tensor_facts(plan.values[index]) for index in plan.outputs],
        "operators": {
            "cortex_m": by_class["stock"],
            "cortex_m_ns": by_class["ns"],
            "portable": by_class["portable"],
        },
        "edge_ops": dict(result.edge_ops),
    }


def sidecar_path(pte_path: Path | str) -> Path:
    return Path(f"{pte_path}.json")


def write_sidecar(result: Any, pte_path: Path | str, kernel_provider: str) -> Path:
    pte_path = Path(pte_path)
    manifest = build_manifest(result, pte_path.read_bytes(), kernel_provider)
    path = sidecar_path(pte_path)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_sidecar(pte_path: Path | str, verify: bool = True) -> dict[str, Any]:
    """Load and (by default) SHA-verify the sidecar for a PTE."""
    pte_path = Path(pte_path)
    path = sidecar_path(pte_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unknown sidecar schema {manifest.get('schema')!r}")
    if verify:
        actual = hashlib.sha256(pte_path.read_bytes()).hexdigest()
        if manifest.get("pte_sha256") != actual:
            raise ValueError(
                f"{path}: sidecar describes PTE {manifest.get('pte_sha256')}, "
                f"but {pte_path.name} is {actual}. Re-export the model."
            )
    return manifest


def lowering_report(manifest: dict[str, Any]) -> str:
    """Human-readable lowering report with fallback severity called out."""
    lines = [
        f"provider: {manifest['kernel_provider']}"
        + ("  (requires NSX_EXECUTORCH_ENABLE_NS_OPS=ON)" if manifest["requires_ns_ops"] else ""),
        f"planned arena: {manifest['planned_arena_size']} bytes",
    ]
    for tensor, kind in [(t, "input") for t in manifest["inputs"]] + [
        (t, "output") for t in manifest["outputs"]
    ]:
        lines.append(f"{kind}: shape={tensor['shape']} dtype={tensor['dtype']} "
                     f"bytes={tensor['size_bytes']}")
    ops = manifest["operators"]
    for name in ops["cortex_m"]:
        lines.append(f"  [stock int8 ] {name}")
    for name in ops["cortex_m_ns"]:
        lines.append(f"  [ns kernel  ] {name}")
    for name in ops["portable"]:
        lines.append(
            f"  [PORTABLE   ] {name}  <- runs on the standard float kernels; "
            "expect a large per-op penalty"
        )
    if not ops["portable"]:
        lines.append("no portable fallbacks: every operator runs on int8 CMSIS-NN kernels")
    return "\n".join(lines)
