"""把完整文本记录划分、编码并写成连续的 token bin。"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import random
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import Config
from dataset.storage import load_dataset

CONFIG_PATH = PROJECT_ROOT / "configs" / "build_bin.json"

EOS_TOKEN = "<|endoftext|>"
SHUFFLE_BLOCK_RECORDS = 1000
BATCH_RECORDS = 512
TOKENIZER_THREADS = 1

_worker_tokenizer = None
_worker_eos_id: int | None = None
_worker_dtype: np.dtype | None = None


# 1. 输入与文件信息
def tokenizer_fingerprint(path: Path) -> str:
    target = path / "tokenizer.json" if path.is_dir() else path
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_dtype(requested: str, tokenizer_size: int) -> np.dtype:
    if requested == "auto":
        requested = "uint16" if tokenizer_size <= 65536 else "uint32"
    if requested not in {"uint16", "uint32"}:
        raise ValueError("dtype must be one of: auto, uint16, uint32.")
    dtype = np.dtype(requested)
    if tokenizer_size - 1 > np.iinfo(dtype).max:
        raise ValueError(f"Tokenizer size {tokenizer_size} does not fit in {dtype}.")
    return dtype


def metadata_path(output_bin: Path) -> Path:
    return output_bin.with_suffix(output_bin.suffix + ".meta.json")


def temporary_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_output_files(train_bin: Path, validation_bin: Path, overwrite: bool) -> None:
    outputs = (train_bin, validation_bin, metadata_path(train_bin), metadata_path(validation_bin))
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Build-bin output already exists. Set overwrite=true in "
            "configs/build_bin.json to replace it: " + ", ".join(map(str, existing))
        )


def load_input_dataset(input_dataset: Path):
    """读取统一中间格式；编码阶段只使用其中的 text 列。"""
    return load_dataset(input_dataset)


# 2. 按记录划分，再打乱块顺序和块内顺序
def split_sizes(total_records: int, train_ratio: float) -> tuple[int, int]:
    if total_records < 2:
        raise ValueError("At least two records are required for train/val splitting.")
    train_records = int(total_records * train_ratio + 0.5)
    train_records = min(max(train_records, 1), total_records - 1)
    return train_records, total_records - train_records


def iter_shuffled_record_batches(
    dataset: Any, train_ratio: float, seed: int,
    shuffle_block_records: int, batch_records: int,
) -> Iterator[tuple[list[str], list[bool]]]:
    total_records = len(dataset)
    train_records, validation_records = split_sizes(total_records, train_ratio)

    # 只抽取较小的一组；mask 中 1 表示验证，0 表示训练。
    sample_validation = validation_records <= train_records
    validation_mask = bytearray(total_records) if sample_validation else bytearray([1]) * total_records
    for index in random.Random(seed).sample(range(total_records), min(train_records, validation_records)):
        validation_mask[index] = int(sample_validation)

    # 划分和打乱使用独立 RNG，保持原有随机调用顺序。
    shuffle_rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
    block_starts = list(range(0, total_records, shuffle_block_records))
    shuffle_rng.shuffle(block_starts)
    texts, validation_flags = [], []
    for block_start in block_starts:
        block_end = min(block_start + shuffle_block_records, total_records)
        block_texts = list(dataset[block_start:block_end]["text"])
        block_offsets = list(range(len(block_texts)))
        shuffle_rng.shuffle(block_offsets)
        for offset in block_offsets:
            text = block_texts[offset]
            if not isinstance(text, str) or not text:
                raise ValueError("The intermediate dataset must contain nonempty strings only.")
            texts.append(text)
            validation_flags.append(bool(validation_mask[block_start + offset]))
            if len(texts) == batch_records:
                yield texts, validation_flags
                texts, validation_flags = [], []
    if texts:
        yield texts, validation_flags


# 3. 每批分词，追加 EOS，分别拼接训练与验证 token
def init_worker(tokenizer_path: str, eos_id: int, dtype_name: str, tokenizer_threads: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from tokenizer import Tokenizer

    global _worker_tokenizer, _worker_eos_id, _worker_dtype
    _worker_tokenizer = Tokenizer(tokenizer_path)
    _worker_eos_id = eos_id
    _worker_dtype = np.dtype(dtype_name)


def encode_batch(payload: tuple[list[str], list[bool]]) -> tuple[np.ndarray, np.ndarray, int, int]:
    texts, validation_flags = payload
    encoded_records = _worker_tokenizer.tokenizer(
        texts, add_special_tokens=False, padding=False, truncation=False,
        return_attention_mask=False, return_token_type_ids=False,
    )["input_ids"]
    train_chunks, validation_chunks = [], []
    for token_ids, is_validation in zip(encoded_records, validation_flags, strict=True):
        token_ids.append(_worker_eos_id)
        chunks = validation_chunks if is_validation else train_chunks
        chunks.append(np.asarray(token_ids, dtype=_worker_dtype))

    empty = np.empty(0, dtype=_worker_dtype)
    return (
        np.concatenate(train_chunks) if train_chunks else empty,
        np.concatenate(validation_chunks) if validation_chunks else empty,
        len(train_chunks),
        len(validation_chunks),
    )


def iter_encoded_batches(
    payloads: Iterator[tuple[list[str], list[bool]]],
    tokenizer_path: Path, eos_id: int, dtype: np.dtype,
    workers: int, tokenizer_threads: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, int, int]]:
    worker_args = (str(tokenizer_path), eos_id, dtype.name, tokenizer_threads)
    if workers == 1:
        init_worker(*worker_args)
        yield from map(encode_batch, payloads)
        return

    # spawn 避免继承分词器/PyTorch 线程；按提交顺序取回，输出不受完成先后影响。
    pending = deque()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker, initargs=worker_args,
    ) as executor:
        for payload in payloads:
            pending.append(executor.submit(encode_batch, payload))
            if len(pending) >= workers * 2:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


# 4. 两份 bin 共用写入、统计、元数据和临时文件处理
def build_bins(
    dataset: Any, input_dataset: Path,
    train_bin: Path, validation_bin: Path,
    tokenizer_path: Path, tokenizer_size: int, tokenizer_sha256: str,
    eos_token: str, eos_id: int, dtype: np.dtype,
    train_ratio: float, seed: int, shuffle_block_records: int, batch_records: int,
    workers: int, tokenizer_threads: int, overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if train_bin.resolve() == validation_bin.resolve():
        raise ValueError("train_bin and val_bin must be different paths.")
    if workers < 1 or tokenizer_threads < 1:
        raise ValueError("workers and tokenizer_threads must be positive.")
    check_output_files(train_bin, validation_bin, overwrite)

    bin_paths = (train_bin, validation_bin)
    final_paths = (*bin_paths, *(metadata_path(path) for path in bin_paths))
    temporary_paths = tuple(temporary_path(path) for path in final_paths)
    for path in bin_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in temporary_paths:
        path.unlink(missing_ok=True)

    totals = [{"records": 0, "tokens": 0}, {"records": 0, "tokens": 0}]
    payloads = iter_shuffled_record_batches(
        dataset, train_ratio, seed, shuffle_block_records, batch_records,
    )
    results = iter_encoded_batches(
        payloads, tokenizer_path, eos_id, dtype, workers, tokenizer_threads,
    )
    try:
        with (
            temporary_paths[0].open("wb") as train_stream,
            temporary_paths[1].open("wb") as validation_stream,
            tqdm(total=len(dataset), unit=" records", desc="Encoding text records") as progress,
        ):
            streams = (train_stream, validation_stream)
            for train_array, validation_array, train_count, validation_count in results:
                for stream, array, count, total in zip(
                    streams, (train_array, validation_array),
                    (train_count, validation_count), totals,
                ):
                    array.tofile(stream)
                    total["records"] += count
                    total["tokens"] += int(array.size)
                progress.update(train_count + validation_count)

        common_metadata = {
            "input_dataset": str(input_dataset.resolve()),
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "input_records": len(dataset),
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
        metadata = []
        for split, path, total, temporary_meta in zip(
            ("train", "validation"), bin_paths, totals, temporary_paths[2:],
        ):
            value = {
                **common_metadata, "split": split, "output_bin": str(path.resolve()),
                "records": total["records"], "tokens": total["tokens"],
            }
            write_json(temporary_meta, value)
            metadata.append(value)

        # 顺序保持为 train.bin、val.bin、train.meta、val.meta。
        for temporary, final in zip(temporary_paths, final_paths):
            temporary.replace(final)
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise

    for path, value in zip(bin_paths, metadata):
        print(
            f"Wrote {value['tokens']:,} {value['split']} tokens from "
            f"{value['records']:,} records to {path.resolve()} ({dtype.name})"
        )
    return metadata[0], metadata[1]


# 5. 配置入口：定位输入，加载分词器，调用构建
def main() -> tuple[dict[str, Any], dict[str, Any]]:
    root_config = Config(CONFIG_PATH)
    config = root_config.require("build_bin")
    input_dataset = root_config.resolve_path("build_bin", "input")
    tokenizer_path = root_config.resolve_path("build_bin", "tokenizer")
    train_bin = root_config.resolve_path("build_bin", "train_bin")
    validation_bin = root_config.resolve_path("build_bin", "val_bin")
    train_ratio, seed = config["train_ratio"], config["seed"]
    workers, overwrite = config["workers"], config["overwrite"]

    # 已有输出时先退出，避免无谓加载数据。
    check_output_files(train_bin, validation_bin, overwrite)
    if not input_dataset.exists():
        raise FileNotFoundError(
            f"Intermediate dataset does not exist: {input_dataset}"
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {tokenizer_path}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if workers == "auto":
        workers = min(4, os.cpu_count() or 1)
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer or 'auto'")

    from tokenizer import Tokenizer

    dataset = load_input_dataset(input_dataset)
    tokenizer = Tokenizer(str(tokenizer_path))
    tokenizer_size = len(tokenizer.tokenizer)
    eos_id = tokenizer.special_token_to_id.get(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer has no special token {EOS_TOKEN!r}")

    return build_bins(
        dataset=dataset, input_dataset=input_dataset,
        train_bin=train_bin, validation_bin=validation_bin,
        tokenizer_path=tokenizer_path, tokenizer_size=tokenizer_size,
        tokenizer_sha256=tokenizer_fingerprint(tokenizer_path),
        eos_token=EOS_TOKEN, eos_id=eos_id, dtype=choose_dtype("auto", tokenizer_size),
        train_ratio=train_ratio, seed=seed,
        shuffle_block_records=SHUFFLE_BLOCK_RECORDS, batch_records=BATCH_RECORDS,
        workers=workers, tokenizer_threads=TOKENIZER_THREADS, overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
