"""Laya runtime, cleaning checkpoints and offline training tools."""
from pathlib import Path

LAYA_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LAYA_ROOT.parents[2]
LABEL_ROOT = LAYA_ROOT.parent
BASE_MODEL_DIRECTORY = LAYA_ROOT / "models" / "laya_multilingual"
CLEANING_MODEL_DIRECTORY = LAYA_ROOT / "models" / "laya_wiki_cleaning_v1"
DATA_DIRECTORY = LAYA_ROOT / "data"
