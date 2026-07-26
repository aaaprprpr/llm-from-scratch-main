import torch

from sft.tokenize_sft import IGNORE_INDEX


class DPOCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def pad_side(self, examples, prefix: str):
        max_len = max(len(example[f"{prefix}_input_ids"]) for example in examples)

        input_ids = []
        labels = []
        attention_mask = []

        for example in examples:
            length = len(example[f"{prefix}_input_ids"])
            pad_len = max_len - length

            input_ids.append(
                example[f"{prefix}_input_ids"] + [self.pad_token_id] * pad_len
            )
            labels.append(example[f"{prefix}_labels"] + [IGNORE_INDEX] * pad_len)
            attention_mask.append(example[f"{prefix}_attention_mask"] + [0] * pad_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __call__(self, examples):
        chosen_input_ids, chosen_labels, chosen_attention_mask = self.pad_side(
            examples, "chosen"
        )
        rejected_input_ids, rejected_labels, rejected_attention_mask = self.pad_side(
            examples, "rejected"
        )

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_attention_mask,
        }
