# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# The HeliaAotBackend delegate blob stored in the PTE. The compiled code is
# not in the PTE: heliaAOT emits C that the firmware links (one NSX module per
# partition), so the blob only names that module and records the delegate's
# tensor contract for the runtime to check. Parsed by
# src/helia_aot_backend.cpp; keep the two in sync.
#
#   offset  size  field
#   0       4     magic "HAOT"
#   4       2     version (1)
#   6       2     module name length N (1..63)
#   8       N     module name (ASCII, no terminator), zero-padded to 4 bytes
#   ...     4     number of inputs I
#   ...     4     number of outputs O
#   ...     8*(I+O) per tensor: u8 ExecuTorch ScalarType, 3 bytes zero, u32 byte size
#
# All integers are little-endian.

from __future__ import annotations

import struct
from dataclasses import dataclass

# ExecuTorch backend id: the runtime BackendInterface and the BackendDetails class share this name.
BACKEND_ID = "HeliaAotBackend"
MAGIC = b"HAOT"
VERSION = 1
# The longest module name the runtime reads (kMaxModuleName in src/helia_aot_backend.cpp).
MAX_MODULE_NAME = 63
# executorch::aten::ScalarType values the delegate exchanges.
SCALAR_TYPES = {"int8": 1, "uint8": 0, "int16": 2, "int32": 3, "float32": 6}


@dataclass(frozen=True)
class TensorContract:
    scalar_type: int
    nbytes: int


@dataclass(frozen=True)
class DelegateBlob:
    module: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]

    def pack(self) -> bytes:
        name = self.module.encode("ascii")
        if not name or len(name) > MAX_MODULE_NAME:
            raise ValueError(
                f"module name {self.module!r} is {len(name)} bytes; the runtime reads 1..{MAX_MODULE_NAME} "
                "ASCII bytes (use a shorter module_prefix)"
            )
        out = bytearray(MAGIC + struct.pack("<HH", VERSION, len(name)) + name)
        out += b"\0" * (-len(out) % 4)
        out += struct.pack("<II", len(self.inputs), len(self.outputs))
        for t in (*self.inputs, *self.outputs):
            out += struct.pack("<B3xI", t.scalar_type, t.nbytes)
        return bytes(out)

    @classmethod
    def unpack(cls, data: bytes) -> "DelegateBlob":
        if data[:4] != MAGIC:
            raise ValueError("not a HeliaAotBackend blob")
        version, name_len = struct.unpack_from("<HH", data, 4)
        if version != VERSION:
            raise ValueError(f"blob version {version}, expected {VERSION}")
        name = data[8:8 + name_len].decode("ascii")
        off = 8 + name_len + (-(8 + name_len) % 4)
        n_in, n_out = struct.unpack_from("<II", data, off)
        off += 8
        tensors = []
        for _ in range(n_in + n_out):
            st, nbytes = struct.unpack_from("<B3xI", data, off)
            tensors.append(TensorContract(st, nbytes))
            off += 8
        return cls(name, tuple(tensors[:n_in]), tuple(tensors[n_in:]))
