# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# The helia-aot delegate's parts that need no heliaAOT: the delegate blob, the
# compile specs and the C module registry. Kept apart from test_helia_aot.py,
# which skips without heliaAOT's PT2E front end, so they run wherever the
# package installs.

import shutil
import subprocess
from pathlib import Path

import pytest

from nsx_cortex_m.helia_aot import DelegateBlob, HeliaAotOptions, TensorContract
from nsx_cortex_m.helia_aot.codegen import compile_specs_for, options_from_specs

REPO = Path(__file__).resolve().parents[2]


def test_blob_round_trip():
    blob = DelegateBlob("haot_0123456789", (TensorContract(1, 490),), (TensorContract(1, 12), TensorContract(6, 48)))
    packed = blob.pack()
    assert packed[:4] == b"HAOT" and len(packed) % 4 == 0
    assert DelegateBlob.unpack(packed) == blob


def test_a_module_name_the_runtime_cannot_read_fails_the_export():
    # src/helia_aot_backend.cpp reads names of up to 63 bytes; a longer one must not reach a device.
    assert DelegateBlob("m" * 63, (), ()).pack()
    with pytest.raises(ValueError, match="shorter module_prefix"):
        DelegateBlob("m" * 64, (), ()).pack()


def test_the_options_travel_in_the_compile_specs(tmp_path):
    # HeliaAotBackend.preprocess() rebuilds its options from the specs alone.
    opts = HeliaAotOptions(output_dir=tmp_path, platform="apollo510_evb", module_prefix="kws",
                           extra={"module": {"schedule": "static"}})
    back = options_from_specs(compile_specs_for(opts))
    assert (back.output_dir, back.platform, back.module_prefix, back.extra) == (
        tmp_path, "apollo510_evb", "kws", {"module": {"schedule": "static"}})


def test_the_registry_counts_the_modules_it_turned_away(tmp_path):
    # Modules register from constructors, where nothing can report a failure: the registry keeps the count for
    # HeliaAotBackend to report when a PTE names a module it cannot find.
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no host C compiler")
    main = tmp_path / "main.c"
    main.write_text(
        '#include "nsx_helia_aot.h"\n'
        "int main(void) {\n"
        '    static const nsx_helia_aot_model_t a = {.name = "a"}, b = {.name = "b"}, c = {.name = "c"};\n'
        "    if (nsx_helia_aot_register(&a) != 0 || nsx_helia_aot_register(&b) != 0) return 1;\n"
        "    if (nsx_helia_aot_register(&a) != 0 || nsx_helia_aot_dropped() != 0) return 2;\n"
        "    if (nsx_helia_aot_register(&c) != -1 || nsx_helia_aot_dropped() != 1) return 3;\n"
        '    if (nsx_helia_aot_find("c") != 0 || nsx_helia_aot_capacity() != 2) return 4;\n'
        "    return 0;\n"
        "}\n"
    )
    exe = tmp_path / "registry"
    subprocess.run([
        cc, "-std=c11", "-DNSX_HELIA_AOT_MAX_MODELS=2", f"-I{REPO / 'includes-api'}", str(main),
        str(REPO / "src" / "nsx_helia_aot_registry.c"), "-o", str(exe),
    ], check=True)
    assert subprocess.run([str(exe)]).returncode == 0
