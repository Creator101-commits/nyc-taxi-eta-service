"""Smoke kernel wrapper: sets NYC_ETA_SMOKE=1 and runs the real entrypoint.

The full entrypoint lives in ``kaggle/run_kaggle.py`` inside the pinned
source dataset; this wrapper only selects smoke mode so both kernels share
one code path.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")


def _find_entrypoint() -> Path:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        entry = candidate.parent / "kaggle" / "run_kaggle.py"
        if (candidate.parent / "src" / "nyc_eta").is_dir() and entry.exists():
            return entry
    raise FileNotFoundError("kaggle/run_kaggle.py not found in source dataset")


def main() -> None:
    os.environ["NYC_ETA_SMOKE"] = "1"
    entry = _find_entrypoint()
    print(f"smoke entrypoint: {entry}", flush=True)
    sys.argv = [str(entry)]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
