"""MiniMind pretrain JSONL -> existing tokenizer -> train/val bins, without cleaning.

Run from the project root: python data_pipeline/build_minimind_bin.py
Configuration: configs/data_pipeline.json -> minimind_bin.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from array import array
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or Path.cwd()).resolve() != SCRIPT_DIR]
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import Config
from data_pipeline.build_bin import (
    build_bins,
    choose_dtype,
    metadata_path,
    tokenizer_fingerprint,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "data_pipeline.json"
PRETRAIN_FILES = {"pretrain_t2t_mini.jsonl", "pretrain_t2t.jsonl"}


def read_text(line: bytes, path: Path, line_number: int) -> str:
    """Validate the schema; never strip, repair, filter or reformat the text."""
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON at {path}:{line_number}") from exc
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        raise ValueError(f"Expected a string 'text' field at {path}:{line_number}")
    if record["text"] == "":
        raise ValueError(f"Empty 'text' at {path}:{line_number}; input was not modified")
    return record["text"]


class JsonlTextDataset:
    """Index line offsets only, so the corpus need not fit in memory or Arrow cache."""

    def __init__(self, path: Path, expected_sha256: str | None = None):
        self.path = Path(path)
        self.offsets = array("Q")
        digest = hashlib.sha256()
        offset = 0
        with self.path.open("rb") as stream, tqdm(
            total=self.path.stat().st_size, unit="B", unit_scale=True,
            desc="Checking MiniMind JSONL",
        ) as progress:
            for line_number, line in enumerate(stream, start=1):
                read_text(line, self.path, line_number)
                self.offsets.append(offset)
                digest.update(line)
                offset += len(line)
                progress.update(len(line))
        self._fingerprint = digest.hexdigest()
        if expected_sha256 and self._fingerprint != expected_sha256:
            raise ValueError(
                f"SHA256 mismatch for {self.path}: expected {expected_sha256}, "
                f"got {self._fingerprint}. Check the file/revision; no bins were written."
            )
        if len(self) < 2:
            raise ValueError("At least two JSONL records are required for train/val splitting")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, key: slice) -> dict[str, list[str]]:
        # The shared encoder requests contiguous record blocks, then shuffles them.
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("Only contiguous record slices are supported")
        texts = []
        if start < stop:
            with self.path.open("rb") as stream:
                stream.seek(self.offsets[start])
                for index in range(start, stop):
                    texts.append(read_text(stream.readline(), self.path, index + 1))
        return {"text": texts}


def configure_hf_cache() -> Path:
    # Run before importing the tokenizer too: transformers imports HF Hub.
    cache = PROJECT_ROOT / "data_pipeline" / "data" / ".cache" / "minimind_hf"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    return cache


def obtain_jsonl(config: dict[str, Any]) -> Path:
    if config["repo_id"] != "jingyaogong/minimind_dataset":
        raise ValueError("This entry point only supports jingyaogong/minimind_dataset")
    if config["filename"] not in PRETRAIN_FILES:
        raise ValueError("Select pretrain_t2t_mini.jsonl or pretrain_t2t.jsonl; SFT/RL are not pretraining inputs")
    raw_dir = PROJECT_ROOT / config["raw_dir"]
    path = raw_dir / config["filename"]
    if path.is_file():
        print(f"Using local JSONL (SHA256 will be verified): {path}")
        return path
    if not config["download"]:
        raise FileNotFoundError(f"Place {config['filename']} in {raw_dir}, or set minimind_bin.download=true")

    # Keep download state inside the project; local_dir avoids a second full copy.
    cache = configure_hf_cache()
    from huggingface_hub import hf_hub_download

    print(f"Downloading only {config['repo_id']}/{config['filename']} @ {config['revision']}")
    return Path(hf_hub_download(
        repo_id=config["repo_id"], repo_type="dataset", filename=config["filename"],
        revision=config["revision"], local_dir=str(raw_dir), cache_dir=str(cache / "hub"),
    ))


def run(config: dict[str, Any]) -> tuple[dict, dict] | None:
    if not config["enabled"]:
        print("MiniMind bin generation is disabled.")
        return None
    if not 0 < config["train_ratio"] < 1:
        raise ValueError("minimind_bin.train_ratio must be between 0 and 1")
    for key in ("shuffle_block_records", "batch_records", "tokenizer_threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"minimind_bin.{key} must be a positive integer")
    if not isinstance(config["seed"], int):
        raise ValueError("minimind_bin.seed must be an integer")
    if not isinstance(config["sha256"], str) or len(config["sha256"]) != 64:
        raise ValueError("minimind_bin.sha256 must contain the source file's SHA256")
    workers = config["workers"]
    if workers == "auto":
        workers = min(4, os.cpu_count() or 1)
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("minimind_bin.workers must be a positive integer or 'auto'")

    train_bin = PROJECT_ROOT / config["train_bin"]
    val_bin = PROJECT_ROOT / config["val_bin"]
    if train_bin.resolve() == val_bin.resolve():
        raise ValueError("train_bin and val_bin must be different paths")
    # Fail before a potentially large download when outputs already exist.
    if not config["overwrite"]:
        for path in (train_bin, val_bin, metadata_path(train_bin), metadata_path(val_bin)):
            if path.exists():
                raise FileExistsError(f"{path} already exists; set minimind_bin.overwrite=true to rebuild")

    configure_hf_cache()
    from tokenizer import Tokenizer

    tokenizer_path = PROJECT_ROOT / config["tokenizer"]
    tokenizer = Tokenizer(str(tokenizer_path))
    tokenizer_size = len(tokenizer.tokenizer)
    eos_id = tokenizer.special_token_to_id.get(config["eos_token"])
    if eos_id is None:
        raise ValueError(f"Tokenizer has no special token {config['eos_token']!r}")
    dtype = choose_dtype(config["dtype"], tokenizer_size)
    fingerprint = tokenizer_fingerprint(tokenizer_path)
    del tokenizer

    path = obtain_jsonl(config)
    dataset = JsonlTextDataset(path, expected_sha256=config["sha256"])
    print(f"MiniMind: {len(dataset):,} records; vocab={tokenizer_size}; no cleaning, truncation or padding")
    # Reuse the binary-format writer only. No download.py, preprocess.py or adapter.
    metadata = build_bins(
        dataset=dataset, input_dataset=path,
        train_bin=train_bin, validation_bin=val_bin,
        tokenizer_path=tokenizer_path, tokenizer_size=tokenizer_size,
        tokenizer_sha256=fingerprint,
        eos_token=config["eos_token"], eos_id=eos_id, dtype=dtype,
        train_ratio=config["train_ratio"], seed=config["seed"],
        shuffle_block_records=config["shuffle_block_records"], batch_records=config["batch_records"],
        workers=workers, tokenizer_threads=config["tokenizer_threads"],
        overwrite=config["overwrite"],
    )
    return metadata


def main() -> None:
    run(Config(CONFIG_PATH).require("minimind_bin"))


if __name__ == "__main__":
    main()
