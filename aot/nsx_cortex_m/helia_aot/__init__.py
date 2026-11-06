# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# ExecuTorch delegate that compiles supported subgraphs with heliaAOT, and
# compile_model(), which compiles a whole model with heliaAOT and no
# ExecuTorch runtime (see standalone.py). build() picks between them: a plain
# module when heliaAOT covers the whole model, the delegate otherwise (see
# select.py).
#
#   export(model, inputs, kernel_provider="ns",
#          helia_aot=HeliaAotOptions(output_dir="build/helia_aot"))
#
# writes the PTE plus one heliaAOT NSX module per delegated partition into
# output_dir. Build the firmware with NSX_EXECUTORCH_ENABLE_HELIA_AOT=ON and
# add those modules; see README "heliaAOT delegate". Needs the helia-aot
# package (pip install 'helia-torch[helia-aot]').
#
# Names load lazily, so reading delegate blobs (sidecar manifests) does not
# require heliaAOT.

from importlib import import_module

_EXPORTS = {
    "BACKEND_ID": "blob",
    "DelegateBlob": "blob",
    "TensorContract": "blob",
    "HeliaAotOptions": "codegen",
    "convert_air": "codegen",
    "AirProgram": "standalone",
    "CompiledModel": "standalone",
    "IoTensor": "standalone",
    "NotFullyDelegableError": "standalone",
    "compile_model": "standalone",
    "program_to_air": "standalone",
    "HeliaAotBuild": "select",
    "build": "select",
    "HeliaAotBackend": "backend",
    "HeliaAotPartitioner": "partitioner",
    "UnsupportedNodeError": "frontend",
    "edge_to_air": "frontend",
    "lower_for_frontend": "frontend",
    "unsupported_reason": "frontend",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
