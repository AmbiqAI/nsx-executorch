#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "external/executorch": "3a97429b0ce0c192861fc3e3729fb81432fd22cf",
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


if __name__ == "__main__":
    main()
