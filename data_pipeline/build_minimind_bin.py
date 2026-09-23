"""Compatibility entry point for the MiniMind pretraining bin builder.

Run from the project root: python data_pipeline/build_minimind_bin.py
The implementation lives in data_pipeline.minimind.build_bin.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or Path.cwd()).resolve() != SCRIPT_DIR]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.minimind.build_bin import (  # noqa: E402
    CONFIG_PATH,
    JsonlTextDataset,
    main,
    obtain_jsonl,
    run,
)

if __name__ == "__main__":
    main()
