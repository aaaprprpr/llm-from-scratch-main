"""Encode train/validation text into flat token-id binary files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_worker_tokenizer = None
_worker_eos_id: int | None = None
_worker_dtype = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize train/validation text and write flat binary token files.")
    parser.add_argument("--tokenizer", type=Path, default=Path("bpe/tokenizer"))
    parser.add_argument("--train-text", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--val-text", type=Path, default=Path("data/val.txt"))
    parser.add_argument("--train-bin", type=Path, default=Path("data/train.bin"))
    parser.add_argument("--val-bin", type=Path, default=Path("data/val.bin"))
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--chunk-lines", type=int, default=2000)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Tokenizer worker processes. Use 1 for deterministic debugging.",
    )
    parser.add_argument("--dtype", choices=("auto", "uint16", "uint32"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tokenizer_fingerprint(path: Path) -> str:
    target = path / "tokenizer.json" if path.is_dir() else path
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_dtype(requested: str, tokenizer_size: int) -> np.dtype:
    if requested == "auto":
        requested = "uint16" if tokenizer_size <= np.iinfo(np.uint16).max + 1 else "uint32"
    dtype = np.dtype(requested)
    if tokenizer_size - 1 > np.iinfo(dtype).max:
        raise ValueError(f"Tokenizer size {tokenizer_size} does not fit in {dtype}.")
    return dtype


def init_worker(tokenizer_path: str, eos_id: int, dtype_name: str) -> None:
    from main.tokenizer_optimized import Tokenizer

    global _worker_tokenizer, _worker_eos_id, _worker_dtype
    _worker_tokenizer = Tokenizer(tokenizer_path)
    _worker_eos_id = eos_id
    _worker_dtype = np.dtype(dtype_name)


def encode_chunk(payload: tuple[list[str], int]) -> tuple[np.ndarray, int, int, int]:
    lines, source_bytes = payload
    if _worker_tokenizer is None or _worker_eos_id is None or _worker_dtype is None:
        raise RuntimeError("Tokenizer worker was not initialized.")

    token_ids: list[int] = []
    nonempty_lines = 0
    for raw_line in lines:
        text = raw_line.strip()
        if not text:
            continue
        ids = _worker_tokenizer.encode(text)
        ids.append(_worker_eos_id)
        token_ids.extend(ids)
        nonempty_lines += 1

    if token_ids and max(token_ids) > np.iinfo(_worker_dtype).max:
        raise ValueError(f"Token id {max(token_ids)} does not fit in {_worker_dtype}.")
    return np.asarray(token_ids, dtype=_worker_dtype), source_bytes, len(lines), nonempty_lines


def iter_line_chunks(path: Path, chunk_lines: int) -> Iterator[tuple[list[str], int]]:
    lines: list[str] = []
    source_bytes = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line in stream:
            lines.append(line)
            source_bytes += len(line.encode("utf-8"))
            if len(lines) >= chunk_lines:
                yield lines, source_bytes
                lines = []
                source_bytes = 0
    if lines:
        yield lines, source_bytes


def build_one_bin(
    input_text: Path,
    output_bin: Path,
    tokenizer_path: Path,
    tokenizer_size: int,
    tokenizer_sha256: str,
    eos_token: str,
    eos_id: int,
    dtype: np.dtype,
    chunk_lines: int,
    workers: int,
    overwrite: bool,
) -> None:
    if not input_text.is_file():
        raise FileNotFoundError(f"Input text does not exist: {input_text}")
    if output_bin.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_bin}. Pass --overwrite to replace it.")

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_bin.with_suffix(output_bin.suffix + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    total_tokens = 0
    total_lines = 0
    nonempty_lines = 0
    chunks = iter_line_chunks(input_text, chunk_lines)

    try:
        with temporary_output.open("wb") as output_stream, tqdm(
            total=input_text.stat().st_size,
            unit="B",
            unit_scale=True,
            desc=f"Encoding {input_text.name}",
        ) as progress:
            if workers == 1:
                init_worker(str(tokenizer_path), eos_id, dtype.name)
                results = map(encode_chunk, chunks)
                for array, source_bytes, line_count, kept_count in results:
                    array.tofile(output_stream)
                    total_tokens += int(array.size)
                    total_lines += line_count
                    nonempty_lines += kept_count
                    progress.update(source_bytes)
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=init_worker,
                    initargs=(str(tokenizer_path), eos_id, dtype.name),
                ) as executor:
                    for array, source_bytes, line_count, kept_count in executor.map(encode_chunk, chunks):
                        array.tofile(output_stream)
                        total_tokens += int(array.size)
                        total_lines += line_count
                        nonempty_lines += kept_count
                        progress.update(source_bytes)

        temporary_output.replace(output_bin)
    except BaseException:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    metadata = {
        "input_text": str(input_text.resolve()),
        "output_bin": str(output_bin.resolve()),
        "dtype": dtype.name,
        "tokens": total_tokens,
        "lines": total_lines,
        "nonempty_lines": nonempty_lines,
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_size": tokenizer_size,
        "tokenizer_sha256": tokenizer_sha256,
        "eos_token": eos_token,
        "eos_id": eos_id,
    }
    metadata_path = output_bin.with_suffix(output_bin.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {total_tokens:,} tokens to {output_bin.resolve()} ({dtype.name})")


def main() -> None:
    args = parse_args()
    if args.chunk_lines < 1:
        raise ValueError("--chunk-lines must be positive.")
    if args.workers < 1:
        raise ValueError("--workers must be positive.")

    from main.tokenizer_optimized import Tokenizer

    tokenizer = Tokenizer(str(args.tokenizer))
    tokenizer_size = len(tokenizer.tokenizer)
    eos_id = tokenizer.special_token_to_id.get(args.eos_token)
    if eos_id is None:
        raise ValueError(
            f"Tokenizer has no special token {args.eos_token!r}. "
            f"Available: {sorted(tokenizer.special_token_to_id)}"
        )

    dtype = choose_dtype(args.dtype, tokenizer_size)
    fingerprint = tokenizer_fingerprint(args.tokenizer)
    common = {
        "tokenizer_path": args.tokenizer.resolve(),
        "tokenizer_size": tokenizer_size,
        "tokenizer_sha256": fingerprint,
        "eos_token": args.eos_token,
        "eos_id": eos_id,
        "dtype": dtype,
        "chunk_lines": args.chunk_lines,
        "workers": args.workers,
        "overwrite": args.overwrite,
    }
    build_one_bin(args.val_text, args.val_bin, **common)
    build_one_bin(args.train_text, args.train_bin, **common)


if __name__ == "__main__":
    main()

