"""Split complete text records and encode them into flat token-id binaries."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import Config

CONFIG_PATH = PROJECT_ROOT / "configs" / "data_pipeline.json"

_worker_tokenizer = None
_worker_eos_id: int | None = None
_worker_dtype: np.dtype | None = None


def tokenizer_fingerprint(path: Path) -> str:
    target = path / "tokenizer.json" if path.is_dir() else path
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_dtype(requested: str, tokenizer_size: int) -> np.dtype:
    if requested == "auto":
        requested = (
            "uint16"
            if tokenizer_size <= np.iinfo(np.uint16).max + 1
            else "uint32"
        )
    if requested not in {"uint16", "uint32"}:
        raise ValueError("dtype must be one of: auto, uint16, uint32.")

    dtype = np.dtype(requested)
    if tokenizer_size - 1 > np.iinfo(dtype).max:
        raise ValueError(f"Tokenizer size {tokenizer_size} does not fit in {dtype}.")
    return dtype


def split_sizes(total_records: int, train_ratio: float) -> tuple[int, int]:
    if total_records < 2:
        raise ValueError("At least two records are required for train/val splitting.")
    train_records = int(total_records * train_ratio + 0.5)
    train_records = min(max(train_records, 1), total_records - 1)
    return train_records, total_records - train_records


def init_worker(
    tokenizer_path: str,
    eos_id: int,
    dtype_name: str,
    tokenizer_threads: int,
) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    from tokenizer import Tokenizer

    global _worker_tokenizer, _worker_eos_id, _worker_dtype
    _worker_tokenizer = Tokenizer(tokenizer_path)
    _worker_eos_id = eos_id
    _worker_dtype = np.dtype(dtype_name)


def encode_batch(
    payload: tuple[list[str], list[bool]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    texts, validation_flags = payload
    if _worker_tokenizer is None or _worker_eos_id is None:
        raise RuntimeError("Tokenizer worker was not initialized.")
    if _worker_dtype is None:
        raise RuntimeError("Tokenizer worker dtype was not initialized.")

    train_chunks = []
    validation_chunks = []

    encoded_records = _worker_tokenizer.tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    for token_ids, is_validation in zip(
        encoded_records,
        validation_flags,
        strict=True,
    ):
        token_ids.append(_worker_eos_id)
        encoded = np.asarray(token_ids, dtype=_worker_dtype)
        if is_validation:
            validation_chunks.append(encoded)
        else:
            train_chunks.append(encoded)

    empty = np.empty(0, dtype=_worker_dtype)
    train_tokens = np.concatenate(train_chunks) if train_chunks else empty
    validation_tokens = (
        np.concatenate(validation_chunks) if validation_chunks else empty
    )
    validation_records = len(validation_chunks)
    return (
        train_tokens,
        validation_tokens,
        len(texts) - validation_records,
        validation_records,
    )


def iter_shuffled_record_batches(
    dataset: Any,
    train_ratio: float,
    seed: int,
    shuffle_block_records: int,
    batch_records: int,
) -> Iterator[tuple[list[str], list[bool]]]:
    total_records = len(dataset)
    target_train_records, target_validation_records = split_sizes(
        total_records,
        train_ratio,
    )

    if target_validation_records <= target_train_records:
        validation_mask = bytearray(total_records)
        sampled_indices = random.Random(seed).sample(
            range(total_records),
            target_validation_records,
        )
        for index in sampled_indices:
            validation_mask[index] = 1
    else:
        validation_mask = bytearray(b"\x01") * total_records
        sampled_indices = random.Random(seed).sample(
            range(total_records),
            target_train_records,
        )
        for index in sampled_indices:
            validation_mask[index] = 0
    del sampled_indices

    shuffle_rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
    block_starts = list(range(0, total_records, shuffle_block_records))
    shuffle_rng.shuffle(block_starts)

    texts = []
    validation_flags = []
    emitted_validation_records = 0
    for block_start in block_starts:
        block_end = min(block_start + shuffle_block_records, total_records)
        block_texts = list(dataset[block_start:block_end]["text"])
        block_offsets = list(range(len(block_texts)))
        shuffle_rng.shuffle(block_offsets)

        for offset in block_offsets:
            text = block_texts[offset]
            if not isinstance(text, str) or not text:
                raise ValueError(
                    "The preprocessed dataset must contain nonempty strings only."
                )

            is_validation = bool(validation_mask[block_start + offset])
            emitted_validation_records += is_validation
            texts.append(text)
            validation_flags.append(is_validation)

            if len(texts) == batch_records:
                yield texts, validation_flags
                texts = []
                validation_flags = []

    if texts:
        yield texts, validation_flags
    if emitted_validation_records != target_validation_records:
        raise RuntimeError("Train/validation split did not emit the expected records.")


def iter_encoded_batches(
    payloads: Iterator[tuple[list[str], list[bool]]],
    tokenizer_path: Path,
    eos_id: int,
    dtype: np.dtype,
    workers: int,
    tokenizer_threads: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, int, int]]:
    if workers == 1:
        init_worker(
            str(tokenizer_path),
            eos_id,
            dtype.name,
            tokenizer_threads,
        )
        yield from map(encode_batch, payloads)
        return

    pending: deque[Future] = deque()
    max_pending = workers * 2
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(
            str(tokenizer_path),
            eos_id,
            dtype.name,
            tokenizer_threads,
        ),
    ) as executor:
        for payload in payloads:
            pending.append(executor.submit(encode_batch, payload))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def metadata_path(output_bin: Path) -> Path:
    return output_bin.with_suffix(output_bin.suffix + ".meta.json")


def temporary_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_bins(
    dataset: Any,
    input_dataset: Path,
    train_bin: Path,
    validation_bin: Path,
    tokenizer_path: Path,
    tokenizer_size: int,
    tokenizer_sha256: str,
    eos_token: str,
    eos_id: int,
    dtype: np.dtype,
    train_ratio: float,
    seed: int,
    shuffle_block_records: int,
    batch_records: int,
    workers: int,
    tokenizer_threads: int,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if train_bin.resolve() == validation_bin.resolve():
        raise ValueError("train_bin and val_bin must be different paths.")
    if workers < 1 or tokenizer_threads < 1:
        raise ValueError("workers and tokenizer_threads must be positive.")
    train_meta_path = metadata_path(train_bin)
    validation_meta_path = metadata_path(validation_bin)
    final_outputs = (
        train_bin,
        validation_bin,
        train_meta_path,
        validation_meta_path,
    )
    existing_outputs = [path for path in final_outputs if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Build-bin output already exists. Set overwrite=true in "
            "configs/data_pipeline.json to replace it: "
            + ", ".join(map(str, existing_outputs))
        )

    for output_bin in (train_bin, validation_bin):
        output_bin.parent.mkdir(parents=True, exist_ok=True)

    temporary_train = temporary_path(train_bin)
    temporary_validation = temporary_path(validation_bin)
    temporary_train_meta = temporary_path(train_meta_path)
    temporary_validation_meta = temporary_path(validation_meta_path)
    temporary_outputs = (
        temporary_train,
        temporary_validation,
        temporary_train_meta,
        temporary_validation_meta,
    )
    for path in temporary_outputs:
        if path.exists():
            path.unlink()

    total_records = len(dataset)
    target_train_records, target_validation_records = split_sizes(
        total_records,
        train_ratio,
    )
    train_records = 0
    validation_records = 0
    train_tokens = 0
    validation_tokens = 0

    payloads = iter_shuffled_record_batches(
        dataset,
        train_ratio,
        seed,
        shuffle_block_records,
        batch_records,
    )
    results = iter_encoded_batches(
        payloads,
        tokenizer_path,
        eos_id,
        dtype,
        workers,
        tokenizer_threads,
    )

    try:
        with (
            temporary_train.open("wb") as train_stream,
            temporary_validation.open("wb") as validation_stream,
            tqdm(
                total=total_records,
                unit=" records",
                desc="Encoding preprocessed dataset",
            ) as progress,
        ):
            for (
                train_array,
                validation_array,
                batch_train_records,
                batch_validation_records,
            ) in results:
                train_array.tofile(train_stream)
                validation_array.tofile(validation_stream)
                train_records += batch_train_records
                validation_records += batch_validation_records
                train_tokens += int(train_array.size)
                validation_tokens += int(validation_array.size)
                progress.update(batch_train_records + batch_validation_records)

        if (train_records, validation_records) != (
            target_train_records,
            target_validation_records,
        ):
            raise RuntimeError(
                "Train/validation record counts do not match the requested split."
            )
        if train_tokens == 0 or validation_tokens == 0:
            raise ValueError("Both train and validation outputs must contain tokens.")

        common_metadata = {
            "input_dataset": str(input_dataset.resolve()),
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "input_records": total_records,
            "train_ratio": train_ratio,
            "validation_ratio": round(1.0 - train_ratio, 15),
            "seed": seed,
            "split_method": "seeded_random_record_indices",
            "shuffle": "seeded_block_and_within_block_shuffle",
            "shuffle_block_records": shuffle_block_records,
            "batch_records": batch_records,
            "workers": workers,
            "tokenizer_threads_per_worker": tokenizer_threads,
            "dtype": dtype.name,
            "tokenizer": str(tokenizer_path.resolve()),
            "tokenizer_size": tokenizer_size,
            "tokenizer_sha256": tokenizer_sha256,
            "eos_token": eos_token,
            "eos_id": eos_id,
        }
        train_metadata = {
            **common_metadata,
            "split": "train",
            "output_bin": str(train_bin.resolve()),
            "records": train_records,
            "tokens": train_tokens,
        }
        validation_metadata = {
            **common_metadata,
            "split": "validation",
            "output_bin": str(validation_bin.resolve()),
            "records": validation_records,
            "tokens": validation_tokens,
        }
        write_json(temporary_train_meta, train_metadata)
        write_json(temporary_validation_meta, validation_metadata)

        temporary_train.replace(train_bin)
        temporary_validation.replace(validation_bin)
        temporary_train_meta.replace(train_meta_path)
        temporary_validation_meta.replace(validation_meta_path)
    except BaseException:
        for path in temporary_outputs:
            if path.exists():
                path.unlink()
        raise

    print(
        f"Wrote {train_tokens:,} train tokens from {train_records:,} records "
        f"to {train_bin.resolve()} ({dtype.name})"
    )
    print(
        f"Wrote {validation_tokens:,} validation tokens from "
        f"{validation_records:,} records to {validation_bin.resolve()} "
        f"({dtype.name})"
    )
    return train_metadata, validation_metadata


def main() -> None:
    config = Config(CONFIG_PATH).require("build_bin")
    input_dataset = PROJECT_ROOT / config["input"]
    tokenizer_path = PROJECT_ROOT / config["tokenizer"]
    train_bin = PROJECT_ROOT / config["train_bin"]
    validation_bin = PROJECT_ROOT / config["val_bin"]
    train_ratio = config["train_ratio"]
    seed = config["seed"]
    shuffle_block_records = config["shuffle_block_records"]
    batch_records = config["batch_records"]
    workers = config["workers"]

    if not input_dataset.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset does not exist: {input_dataset}"
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {tokenizer_path}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not isinstance(shuffle_block_records, int) or shuffle_block_records < 1:
        raise ValueError("shuffle_block_records must be positive.")
    if not isinstance(batch_records, int) or batch_records < 1:
        raise ValueError("batch_records must be positive.")

    available_cpus = os.cpu_count() or 2
    if workers == "auto":
        workers = min(4, available_cpus)
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer or 'auto'.")
    tokenizer_threads = max(1, available_cpus // workers)

    from datasets import load_from_disk
    from tokenizer import Tokenizer

    dataset = load_from_disk(str(input_dataset))
    if getattr(dataset, "column_names", None) != ["text"]:
        raise ValueError(
            "The preprocessed dataset must be a Dataset with only a text column."
        )

    tokenizer = Tokenizer(str(tokenizer_path))
    tokenizer_size = len(tokenizer.tokenizer)
    eos_id = tokenizer.special_token_to_id.get(config["eos_token"])
    if eos_id is None:
        raise ValueError(
            f"Tokenizer has no special token {config['eos_token']!r}. "
            f"Available: {sorted(tokenizer.special_token_to_id)}"
        )

    dtype = choose_dtype(config["dtype"], tokenizer_size)
    build_bins(
        dataset=dataset,
        input_dataset=input_dataset,
        train_bin=train_bin,
        validation_bin=validation_bin,
        tokenizer_path=tokenizer_path,
        tokenizer_size=tokenizer_size,
        tokenizer_sha256=tokenizer_fingerprint(tokenizer_path),
        eos_token=config["eos_token"],
        eos_id=eos_id,
        dtype=dtype,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_block_records=shuffle_block_records,
        batch_records=batch_records,
        workers=workers,
        tokenizer_threads=tokenizer_threads,
        overwrite=config["overwrite"],
    )


if __name__ == "__main__":
    main()
