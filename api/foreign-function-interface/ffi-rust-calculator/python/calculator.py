"""Python wrapper. Marshalling only — the logic lives in Rust."""

import ctypes
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SUFFIX = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
_PREFIX = "" if sys.platform == "win32" else "lib"
_LIB_PATH = (
    Path(__file__).resolve().parent.parent
    / "target"
    / "release"
    / f"{_PREFIX}calculator.{_SUFFIX}"
)

_lib = ctypes.CDLL(str(_LIB_PATH))

_lib.calculator_add.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.calculator_add.restype = ctypes.c_int

_lib.calculator_minus.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.calculator_minus.restype = ctypes.c_int


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return _lib.calculator_add(a, b)


def minus(a: int, b: int) -> int:
    """Subtract b from a."""
    return _lib.calculator_minus(a, b)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    a, b = 5, 3

    logger.info(f"{a} + {b} = {add(a, b)}")
    logger.info(f"{a} - {b} = {minus(a, b)}")
