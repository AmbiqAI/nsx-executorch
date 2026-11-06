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
#
# A float32 .pt2 is PT2E-quantized here (--calibrate) before lowering; an
# already-INT8 .pt2 skips quantization and goes straight to kernel matching.
# Both take the same export() path as the Python API.
#
# With the ns provider and heliaAOT installed (the helia-aot extra), compile
# uses heliaAOT by default (--helia-aot auto): a model heliaAOT fully covers
# becomes one plain heliaAOT module with no ExecuTorch runtime, and any other
# model a PTE that delegates the parts heliaAOT supports. --helia-aot module /
# delegate force one path (module fails on a model heliaAOT does not fully
# cover); --helia-aot off lowers to the cortex_m kernels only.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_pt2(path: Path):
    import torch

    # Deserialization resolves quantized_decomposed.* ops for INT8 programs,
    # which only exist once torchao's PT2E op library is registered.
    import torchao.quantization.pt2e  # noqa: F401

    exported = torch.export.load(str(path))
    args, kwargs = exported.example_inputs
    if kwargs:
        raise SystemExit(
            f"{path}: exported program takes keyword inputs {sorted(kwargs)}; "
            "helia-torch supports positional tensor inputs only."
        )
    non_tensor = [
        f"argument {index} ({type(arg).__name__})"
        for index, arg in enumerate(args)
        if not isinstance(arg, torch.Tensor)
    ]
    if non_tensor:
        raise SystemExit(
            f"{path}: exported program takes non-tensor inputs "
            f"({', '.join(non_tensor)}); helia-torch supports positional "
            "tensor inputs only."
        )
    return exported, tuple(args)


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


def _helia_aot_mode(args: argparse.Namespace) -> str:
    """The --helia-aot mode to run: auto, module, delegate or off."""
    import importlib.util

    mode = args.helia_aot
    if mode == "off":
        return mode
    if args.provider != "ns":
        if mode == "auto":
            return "off"
        raise SystemExit(
            f"--helia-aot {mode} needs --provider ns (heliaAOT links ns-cmsis-nn)."
        )
    if importlib.util.find_spec("helia_aot") is None:
        if mode == "auto":
            print(
                "note: heliaAOT is not installed, so the model is lowered to the "
                "cortex_m kernels only (pip install 'helia-torch[helia-aot]' to use it).",
                file=sys.stderr,
            )
            return "off"
        raise SystemExit(
            f"--helia-aot {mode} needs heliaAOT: pip install 'helia-torch[helia-aot]'."
        )
    return mode


def _write_pte(result, output: Path, provider: str) -> None:
    from .manifest import load_sidecar, lowering_report

    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_pte(output)  # sidecar written alongside
    manifest = load_sidecar(output)
    print(f"wrote {output} and {output}.json")
    print(lowering_report(manifest))
    if manifest["operators"]["portable"] and provider == "ns":
        print(
            "note: portable ops above stayed on the standard float kernels "
            "(an NS qualifier declined them); see the ns lowering rules.",
            file=sys.stderr,
        )


def _compile_with_helia_aot(args, exported, example, calibration, mode: str, output: Path) -> int:
    from .helia_aot import HeliaAotOptions, NotFullyDelegableError, build

    options = HeliaAotOptions(output_dir=args.helia_aot_dir or output.parent)
    if args.platform is not None:
        options.platform = args.platform
    try:
        built = build(
            exported,
            example,
            options,
            mode=mode,
            name=args.name,
            calibration_samples=calibration,
            int8_io=args.int8_io,
        )
    except NotFullyDelegableError as exc:
        raise SystemExit(
            f"{args.model}: --helia-aot module needs a model heliaAOT fully covers.\n{exc}\n"
            "Use --helia-aot auto or delegate to run the rest in ExecuTorch."
        ) from None

    if built.mode == "module":
        compiled = built.compiled
        print(
            f"heliaAOT covers the whole model: wrote the module {compiled.module_dir} "
            f"and its I/O contract {compiled.module_dir.parent / (compiled.name + '_io.json')} "
            "(no PTE; the application calls the module directly)"
        )
        if args.int8_io:
            print("note: --int8-io has no effect on a module, whose I/O is always int8.", file=sys.stderr)
        return 0

    if built.rejected:
        print(
            f"heliaAOT does not cover {len(built.rejected)} operator(s), so the model is a PTE "
            "that delegates the rest to heliaAOT:",
            file=sys.stderr,
        )
        for node, reason in built.rejected.items():
            print(f"  {node}: {reason}", file=sys.stderr)
    _write_pte(built.export_result, output, args.provider)
    print(f"heliaAOT modules: {options.output_dir}")
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    from . import export, is_pt2e_quantized

    mode = _helia_aot_mode(args)
    exported, example = _load_pt2(args.model)
    if is_pt2e_quantized(exported):
        if args.calibrate is not None:
            raise SystemExit(
                f"{args.model} is already PT2E-quantized; --calibrate only "
                "applies when helia-torch performs the quantization."
            )
        calibration = None
    else:
        calibration = _calibration(args.calibrate or "random:8", example)
    output = args.output or args.model.with_suffix(".pte")
    if mode != "off":
        return _compile_with_helia_aot(args, exported, example, calibration, mode, output)
    result = export(
        exported,
        example,
        kernel_provider=args.provider,
        calibration_samples=calibration,
        int8_io=args.int8_io,
    )
    _write_pte(result, output, args.provider)
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
        "compile",
        help=(
            "Quantize (if float) and lower a torch.export .pt2 file to a "
            ".pte + sidecar; INT8 .pt2 inputs are lowered as-is"
        ),
    )
    compile_parser.add_argument(
        "model", type=Path, help="Input .pt2 (torch.export.save), float32 or INT8"
    )
    compile_parser.add_argument(
        "--provider",
        choices=("arm", "ns"),
        default="ns",
        help="CMSIS-NN kernel provider to lower for (default: ns)",
    )
    compile_parser.add_argument(
        "--calibrate",
        default=None,
        help=(
            "Quantization calibration for float inputs: 'random:N' or an "
            ".npz of sample batches (default: random:8); invalid for "
            "already-quantized inputs"
        ),
    )
    compile_parser.add_argument(
        "--int8-io",
        action="store_true",
        help=(
            "Serialize the method with int8 tensor I/O (like an int8 TFLite "
            "model) instead of the default float32 boundary; the host must "
            "then quantize inputs / dequantize outputs itself"
        ),
    )
    compile_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .pte path (default: the model's path with .pte); a heliaAOT module "
        "build writes no PTE",
    )
    compile_parser.add_argument(
        "--helia-aot",
        choices=("auto", "module", "delegate", "off"),
        default="auto",
        help=(
            "How heliaAOT runs the model (ns provider). auto (default): a plain heliaAOT "
            "module with no ExecuTorch runtime when heliaAOT covers every operator, "
            "otherwise a PTE delegating the parts it covers; falls back to off when "
            "heliaAOT is not installed. module: always a plain module, an error if "
            "heliaAOT does not cover the model. delegate: always a PTE. off: no heliaAOT."
        ),
    )
    compile_parser.add_argument(
        "--helia-aot-dir",
        type=Path,
        default=None,
        help="Directory for the generated heliaAOT modules (default: the output's directory)",
    )
    compile_parser.add_argument(
        "--name",
        default=None,
        help="C name of a plain heliaAOT module (default: haot_<hash>)",
    )
    compile_parser.add_argument(
        "--platform",
        default=None,
        help="heliaAOT platform to generate for (default: heliaAOT's Apollo510 platform)",
    )
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
