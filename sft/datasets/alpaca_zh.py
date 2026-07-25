from pathlib import Path
from typing import Mapping

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from torch.utils.data import Dataset

from sft.schema import ChatExample, make_message

REQUIRED_COLUMNS = {"instruction", "input", "output"}


def load_alpaca_zh(path: str | Path) -> HFDataset:
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise KeyError("Alpaca 中文数据集中不存在 train split")
        dataset = dataset["train"]

    missing_columns = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise KeyError(f"Alpaca 中文数据集缺少字段：{names}")
    return dataset


def adapt_alpaca_example(example: Mapping[str, object]) -> ChatExample:
    instruction = str(example["instruction"]).strip()
    input_text = str(example.get("input") or "").strip()
    output = str(example["output"]).strip()

    if input_text:
        user_content = f"{instruction}\n\n输入：\n{input_text}"
    else:
        user_content = instruction

    return {
        "messages": [
            make_message("user", user_content),
            make_message("assistant", output),
        ]
    }


class AlpacaZhDataset(Dataset):
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ChatExample:
        return adapt_alpaca_example(self.dataset[index])
