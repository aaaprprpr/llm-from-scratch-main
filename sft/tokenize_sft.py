from typing import TypedDict

IGNORE_INDEX = -100


class TokenizedExample(TypedDict):
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def tokenize_chat_example(
    example, tokenizer, max_seq_len: int
) -> TokenizedExample | None:
    enc = tokenizer.apply_chat_template(
        example["messages"],
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
