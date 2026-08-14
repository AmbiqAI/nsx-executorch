"""Minimal Python 3.12 compatibility surface required by pinned torchgen.

ExecuTorch's build-time code generator imports these names from the external
typing_extensions package even though both are part of Python 3.11+.
"""

from typing import Self, assert_never

__all__ = ["Self", "assert_never"]
