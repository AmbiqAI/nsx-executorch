# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# helia-torch: command-line export tool for the nsx-executorch runtime.
#
# The compile input is a `.pt2` file — the artifact `torch.export.save()`
# produces. That keeps the CLI file-in/file-out (like helia-aot) and avoids
# executing arbitrary model code: the user runs torch.export in their own
# training environment, then hands the serialized program to this tool.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_pt2(path: Path):
    import torch

    exported = torch.export.load(str(path))
    args, kwargs = exported.example_inputs
    if kwargs:
        raise SystemExit(
            f"{path}: exported program takes keyword inputs {sorted(kwargs)}; "
            "helia-torch supports positional tensor inputs only."
        )
    return exported.module(), tuple(args)


def _calibration(spec: str, example: tuple) -> list[tuple]:
    import torch

    if spec.startswith("random:"):
        count = int(spec.split(":", 1)[1])
        samples = []
        for index in range(count):
            generator = torch.Generator().manual_seed(index)
            samples.append(
                tuple(
                    torch.randn(tensor.shape, generator=generator).to(
                        memory_format=(
                            torch.channels_last
                            if tensor.dim() == 4 and not tensor.is_contiguous()
                            else torch.contiguous_format
                        )
                    )
                    for tensor in example
                )
            )
        return samples

    import numpy as np

    path = Path(spec)
    if not path.is_file():
        raise SystemExit(f"calibration file not found: {path}")
    archive = np.load(path)
    keys = sorted(archive.files)
    samples = []
    for key in keys:
        batch = archive[key]
        if batch.shape[1:] != tuple(example[0].shape[1:]):
            raise SystemExit(
                f"{path}:{key} has shape {batch.shape}, expected "
                f"(N, *{tuple(example[0].shape[1:])})"
            )
        for row in batch:
            samples.append((torch.from_numpy(row[None, ...]).float(),))
    if len(example) != 1:
        raise SystemExit("npz calibration currently supports single-input models")
    return samples


def _cmd_compile(args: argparse.Namespace) -> int:
    from . import export

    model, example = _load_pt2(args.model)
    calibration = _calibration(args.calibrate, example)
    result = export(
        model, example, kernel_provider=args.provider, calibration_samples=calibration
    )
    output = args.output or args.model.with_suffix(".pte")
    result.write_pte(output)  # sidecar written alongside

    from .manifest import load_sidecar, lowering_report

    manifest = load_sidecar(output)
    print(f"wrote {output} and {output}.json")
    print(lowering_report(manifest))
    if manifest["operators"]["portable"] and args.provider == "ns":
        print(
            "note: portable ops above stayed on the standard float kernels "
            "(an NS qualifier declined them); see the ns lowering rules.",
            file=sys.stderr,
        )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .manifest import load_sidecar

    manifest = load_sidecar(args.model, verify=not args.no_verify)
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .manifest import load_sidecar, lowering_report

    manifest = load_sidecar(args.model, verify=not args.no_verify)
    print(lowering_report(manifest))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="helia-torch",
        description="Export PyTorch models for the nsx-executorch Cortex-M runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="Quantize and lower a torch.export .pt2 file to a .pte + sidecar"
    )
    compile_parser.add_argument("model", type=Path, help="Input .pt2 (torch.export.save)")
    compile_parser.add_argument(
        "--provider",
        choices=("arm", "ns"),
        default="ns",
        help="CMSIS-NN kernel provider to lower for (default: ns)",
    )
    compile_parser.add_argument(
        "--calibrate",
        default="random:8",
        help="Quantization calibration: 'random:N' or an .npz of sample batches",
    )
    compile_parser.add_argument("-o", "--output", type=Path, help="Output .pte path")
    compile_parser.set_defaults(func=_cmd_compile)

    for name, func, help_text in (
        ("inspect", _cmd_inspect, "Print a compiled model's sidecar manifest as JSON"),
        ("report", _cmd_report, "Print a compiled model's kernel lowering report"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("model", type=Path, help="Compiled .pte (with .pte.json sidecar)")
        sub.add_argument(
            "--no-verify", action="store_true", help="Skip the sidecar/PTE SHA-256 check"
        )
        sub.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
