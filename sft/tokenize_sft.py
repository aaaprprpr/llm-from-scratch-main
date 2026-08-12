import multiprocessing
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict

from sft.utils import load_tokenizer_from_paths

IGNORE_INDEX = -100


class TokenizedExample(TypedDict):
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


_WORKER_TOKENIZER = None
_WORKER_MAX_SEQ_LEN = 0


def tokenize_chat_example(
    example, tokenizer, max_seq_len: int
) -> TokenizedExample | None:
    if example is None:
        return None

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


def resolve_tokenize_workers(value) -> int:
    if value == "auto":
        cpu_count = os.cpu_count() or 1
        return min(8, max(1, cpu_count - 1))

    workers = int(value)
    if workers < 1:
        raise ValueError("data.tokenize_workers 必须是正整数或 'auto'")
    return workers


def _init_tokenize_worker(
    tokenizer_path: str,
    chat_template_path: str,
    max_seq_len: int,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_MAX_SEQ_LEN

    # 每个进程已经独占一个 CPU worker，关闭 tokenizer 内部线程，避免过度抢占。
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = load_tokenizer_from_paths(
        tokenizer_path,
        chat_template_path,
    )
    _WORKER_MAX_SEQ_LEN = max_seq_len


def _init_tokenize_worker_from_parent(tokenizer, max_seq_len: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_MAX_SEQ_LEN

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = tokenizer
    _WORKER_MAX_SEQ_LEN = max_seq_len


def _tokenize_in_worker(example) -> TokenizedExample | None:
    return tokenize_chat_example(
        example,
        _WORKER_TOKENIZER,
        max_seq_len=_WORKER_MAX_SEQ_LEN,
    )


def iter_tokenized_examples(
    examples: Iterable,
    tokenizer,
    max_seq_len: int,
    workers: int,
    tokenizer_path: str | Path,
    chat_template_path: str | Path,
    chunksize: int,
) -> Iterator[TokenizedExample | None]:
    if chunksize < 1:
        raise ValueError("data.tokenize_chunksize 必须是正整数")

    if workers == 1:
        for example in examples:
            yield tokenize_chat_example(
                example,
                tokenizer,
                max_seq_len=max_seq_len,
            )
        return

    if sys.platform.startswith("linux"):
        # train() 会在 CUDA 初始化之前构建 cache；fork 可共享父进程已加载的 tokenizer。
        context = multiprocessing.get_context("fork")
        initializer = _init_tokenize_worker_from_parent
        initargs = (tokenizer, max_seq_len)
    else:
        context = multiprocessing.get_context("spawn")
        initializer = _init_tokenize_worker
        initargs = (str(tokenizer_path), str(chat_template_path), max_seq_len)

    with context.Pool(
        processes=workers,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        yield from pool.imap(
            _tokenize_in_worker,
            examples,
            chunksize=chunksize,
        )
