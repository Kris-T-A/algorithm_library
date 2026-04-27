"""Test fixtures shared across torch_modules tests.

Inserts the python_algorithm_library .pyd location into sys.path so alignment
tests can `import PythonAlgorithmLibrary`. If the build artifact is missing,
alignment tests skip with a clear message.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PYD_DIR = REPO_ROOT / "libs" / "python_algorithm_library" / "python_script"
sys.path.insert(0, str(PYD_DIR))


@pytest.fixture(scope="session")
def pal():
    """The C++ oracle (`PythonAlgorithmLibrary`). Skips the test if the pyd is missing."""
    try:
        import PythonAlgorithmLibrary as _pal
    except ImportError as exc:
        pytest.skip(
            f"PythonAlgorithmLibrary not importable from {PYD_DIR}: {exc}. "
            "Build the python_algorithm_library target first."
        )
    return _pal
