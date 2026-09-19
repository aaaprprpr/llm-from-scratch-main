import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.stdout.reconfigure(encoding="utf-8")
from label.backend.llm_cleaning import CleaningConfig, LlmCleaner
from label.backend.dataset_store import load_dataset
from label.backend.blocks import parse_blocks

directory = Path(__file__).parent
cleaner = LlmCleaner(CleaningConfig.from_file(), directory / "suggestions")
samples = [
    ("句内垃圾", [{"id": "a", "text": "猫是一种哺乳动物（点击领取优惠券），喜欢晒太阳。", "separator_after": ""}]),
    ("断行拼接", [{"id": "a", "text": "猫是一种哺乳动", "separator_after": "\n\n"},
                  {"id": "b", "text": "物，喜欢晒太阳。", "separator_after": "\n\n"},
                  {"id": "c", "text": "点击注册领取优惠券", "separator_after": ""}]),
]
for title, blocks in samples:
    result = cleaner.clean(blocks, title=title, provenance={"test": title})
    print(json.dumps({"sample": title, "result": result}, ensure_ascii=False), flush=True)
