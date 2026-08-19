#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "external/executorch": "3a97429b0ce0c192861fc3e3729fb81432fd22cf",
}
PROVIDER_PINS = {
    "arm-cmsis-nn": "6d21a6f821fb72541173a6c4d05d83329fa74f7c",
    "nsx-cmsis-nn": "631726420b04860a5c4236956a3741ff5a96bd7f",
}


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True
    ).strip()


def main() -> None:
    for relative, expected in PINS.items():
        checkout = ROOT / relative
        actual = git("rev-parse", "HEAD", cwd=checkout)
        assert actual == expected, f"{relative}: expected {expected}, found {actual}"
        # Nested submodules are separate pinned gitlinks. Ignore their hydrated
        # worktree state while still rejecting edits to this source checkout.
        dirty = git("status", "--porcelain", "--ignore-submodules=dirty", cwd=checkout)
        assert not dirty, f"{relative} contains local source modifications:\n{dirty}"

    source = ROOT / "external/executorch"
    forbidden = [
        "EXECUTORCH_CORTEX_M_USE_NS_CMSIS_NN",
        "NSX_EXECUTORCH",
    ]
    tracked_text = "\n".join(
        (source / path).read_text(encoding="utf-8", errors="ignore")
        for path in [
            "backends/cortex_m/ops/operators.yaml",
            "backends/cortex_m/ops/op_quantized_conv2d.cpp",
            "backends/cortex_m/passes/convert_to_cortex_m_pass.py",
        ]
    )
    for marker in forbidden:
        assert marker not in tracked_text, f"stock ExecuTorch contains {marker}"

    manifest = (ROOT / "nsx-module.yaml").read_text(encoding="utf-8")
    version = (ROOT / "version.txt").read_text(encoding="utf-8").strip()
    assert f'version: "{version}"' in manifest

    # CMSIS-NN and CMSIS_6 must never come back as vendored submodules; both
    # providers are resolved as NSX module dependencies instead (see
    # PROVENANCE.md / README.md).
    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    for forbidden_submodule in ["CMSIS-NN", "CMSIS_6"]:
        assert forbidden_submodule not in gitmodules, (
            f".gitmodules must not reference {forbidden_submodule}"
        )
    assert not (ROOT / "external/CMSIS-NN").exists()
    assert not (ROOT / "external/CMSIS_6").exists()

    provenance = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    for module, revision in PROVIDER_PINS.items():
        assert module in provenance
        assert revision in provenance

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "FETCHCONTENT_FULLY_DISCONNECTED ON" in cmake
    assert "cmake/cmsis-nn-provider" in cmake
    assert "nsx_cmsis_nn_compat.h" in cmake
    # NS additional kernels stay opt-in and out-of-tree: the gating flag must
    # exist and the cortex_m_ns operator schemas must live in this repo (the
    # pinned ExecuTorch tree is never modified).
    assert "NSX_EXECUTORCH_ENABLE_NS_OPS" in cmake
    assert "ops-ns/operators_ns.yaml" in cmake

    required_optional = {"arm-cmsis-nn", "nsx-cmsis-nn"}
    optional_block = manifest.split("optional:", 1)[1]
    for module in required_optional:
        assert f"- {module}" in optional_block


if __name__ == "__main__":
    main()
