from pathlib import Path
from typing import Mapping

from datasets import Dataset as HFDataset

from train.sft.datasets.instruction import (
    InstructionDataset,
    adapt_instruction_example,
    load_instruction_dataset,
)
from train.sft.schema import ChatExample


def load_alpaca_zh(path: str | Path) -> HFDataset:
    return load_instruction_dataset(path, "Alpaca 中文数据集")


def adapt_alpaca_example(example: Mapping[str, object]) -> ChatExample | None:
    return adapt_instruction_example(example)


class AlpacaZhDataset(InstructionDataset):
    pass
