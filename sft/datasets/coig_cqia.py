from pathlib import Path
from typing import Mapping

from datasets import Dataset as HFDataset

from sft.datasets.instruction import (
    InstructionDataset,
    adapt_instruction_example,
    load_instruction_dataset,
)
from sft.schema import ChatExample


def load_coig_cqia(path: str | Path) -> HFDataset:
    return load_instruction_dataset(path, "COIG-CQIA 数据集")


def adapt_coig_cqia_example(example: Mapping[str, object]) -> ChatExample | None:
    return adapt_instruction_example(example)


class COIGCQIADataset(InstructionDataset):
    pass
