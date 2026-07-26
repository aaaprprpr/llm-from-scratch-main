from typing import TypedDict

from sft.tokenize_sft import IGNORE_INDEX


class TokenizedDPOExample(TypedDict):
    chosen_input_ids: list[int]
    chosen_labels: list[int]
    chosen_attention_mask: list[int]
    rejected_input_ids: list[int]
    rejected_labels: list[int]
    rejected_attention_mask: list[int]


def tokenize_messages(messages, tokenizer, max_seq_len: int):
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )

    full_input_ids = enc["input_ids"]
    full_assistant_mask = enc["assistant_masks"]

    if any(full_assistant_mask[max_seq_len:]):
        return None

    input_ids = full_input_ids[:max_seq_len]
    assistant_mask = full_assistant_mask[:max_seq_len]

    if not any(assistant_mask):
        return None

    labels = [
        token_id if is_assistant else IGNORE_INDEX
        for token_id, is_assistant in zip(input_ids, assistant_mask)
    ]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def tokenize_preference_example(
    example, tokenizer, max_seq_len: int
) -> TokenizedDPOExample | None:
    chosen = tokenize_messages(example["chosen_messages"], tokenizer, max_seq_len)
    rejected = tokenize_messages(example["rejected_messages"], tokenizer, max_seq_len)

    if chosen is None or rejected is None:
        return None

    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_labels": chosen["labels"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_labels": rejected["labels"],
        "rejected_attention_mask": rejected["attention_mask"],
    }
