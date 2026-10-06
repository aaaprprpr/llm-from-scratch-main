"""本地 Dataset/Arrow 中间格式的读取接口。"""

from __future__ import annotations

import json
from pathlib import Path

ARROW_STORE_VERSION = 1
DESCRIPTOR_NAME = "dataset.json"


def load_dataset(path: str | Path):
    """Open either this one-pass Arrow layout or HF ``save_to_disk`` output."""

    from datasets import Dataset, concatenate_datasets, load_from_disk

    directory = Path(path).resolve()
    descriptor_path = directory / DESCRIPTOR_NAME
    if not descriptor_path.is_file():
        return load_from_disk(str(directory))
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("store_version") != ARROW_STORE_VERSION:
        raise ValueError(
            f"Unsupported Arrow store version: {descriptor.get('store_version')!r}"
        )
    files = descriptor.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Arrow store descriptor contains no files")
    datasets = []
    for relative_path in files:
        file_path = directory / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        datasets.append(Dataset.from_file(str(file_path)))
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    if len(dataset) != int(descriptor["records"]):
        raise ValueError("Arrow store descriptor row count mismatch")
    dataset._fingerprint = str(descriptor["fingerprint"])
    return dataset
