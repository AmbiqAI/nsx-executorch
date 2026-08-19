# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

# Make the in-repo `nsx_cortex_m` package and test helpers importable.
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[0]))
sys.path.insert(0, str(_here))

# These tests need a host ExecuTorch (+ torch) install; skip cleanly when the
# AOT environment is not available (e.g. firmware-only CI jobs).
pytest.importorskip("executorch")
pytest.importorskip("executorch.backends.cortex_m.passes")
