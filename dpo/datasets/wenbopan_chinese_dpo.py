from pathlib import Path
from typing import Mapping

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset, load_from_disk

from chat_schema import ChatMessage, make_message
from dpo.schema import PreferenceExample

DATASET_NAME = "wenbopan/Chinese-dpo-pairs"
REQUIRED_COLUMNS = {"prompt", "chosen", "rejected"}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_wenbopan_chinese_dpo(
    source: str | Path = DATASET_NAME,
    split: str = "train",
) -> HFDataset:
    path = Path(source)
    if path.exists():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise KeyError(f"DPO 数据集中不存在 split：{split}")
            dataset = dataset[split]
    else:
        dataset = load_dataset(str(source), split=split)

    missing_columns = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise KeyError(f"DPO 数据集缺少字段：{names}")
    return dataset


def is_valid_wenbopan_example(example: Mapping[str, object]) -> bool:
    return all(normalize_text(example.get(key)) for key in REQUIRED_COLUMNS)


def build_prompt_messages(example: Mapping[str, object]) -> list[ChatMessage]:
    messages = []
    system = normalize_text(example.get("system"))
    if system and system.lower() not in {"none", "null"}:
        messages.append(make_message("system", system))
    messages.append(make_message("user", normalize_text(example["prompt"])))
    return messages


def adapt_wenbopan_example(example: Mapping[str, object]) -> PreferenceExample:
    prompt_messages = build_prompt_messages(example)
    chosen = normalize_text(example["chosen"])
    rejected = normalize_text(example["rejected"])

    return {
        "chosen_messages": prompt_messages + [make_message("assistant", chosen)],
        "rejected_messages": prompt_messages + [make_message("assistant", rejected)],
    }


class WenbopanChineseDPODataset:
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset
        self.indices = [
            index
            for index in range(len(dataset))
            if is_valid_wenbopan_example(dataset[index])
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> PreferenceExample:
        return adapt_wenbopan_example(self.dataset[self.indices[index]])
