from pathlib import Path
from typing import Mapping

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from torch.utils.data import Dataset

from sft.schema import ChatExample, make_message

REQUIRED_COLUMNS = {"instruction", "input", "output"}


def load_instruction_dataset(path: str | Path, dataset_name: str) -> HFDataset:
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise KeyError(f"{dataset_name} 中不存在 train split")
        dataset = dataset["train"]

    missing_columns = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise KeyError(f"{dataset_name} 缺少字段：{names}")
    return dataset


def adapt_instruction_example(
    example: Mapping[str, object],
) -> ChatExample | None:
    instruction = str(example.get("instruction") or "").strip()
    input_text = str(example.get("input") or "").strip()
    output = str(example.get("output") or "").strip()

    if not instruction or not output:
        return None

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


class InstructionDataset(Dataset):
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ChatExample | None:
        return adapt_instruction_example(self.dataset[index])
