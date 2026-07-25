import torch

from sft.tokenize_sft import IGNORE_INDEX


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        max_len = max(len(example["input_ids"]) for example in examples)

        input_ids = []
        labels = []
        attention_mask = []

        for example in examples:
            length = len(example["input_ids"])
            pad_len = max_len - length

            input_ids.append(example["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(example["labels"] + [IGNORE_INDEX] * pad_len)
            attention_mask.append(example["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
