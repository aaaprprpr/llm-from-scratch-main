from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ARROW_STORE_VERSION = 1
DESCRIPTOR_NAME = "dataset.json"


@dataclass(frozen=True)
class ArrowStoreResult:
    records: int
    bytes_written: int
    fingerprint: str
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _maximum_bytes(value: str | int) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("max_shard_size must be positive")
        return value
    from datasets.utils.py_utils import convert_file_size_to_int

    result = int(convert_file_size_to_int(value))
    if result <= 0:
        raise ValueError("max_shard_size must be positive")
    return result


def write_arrow_dataset(
    rows: Iterable[Mapping[str, Any]],
    output_directory: str | Path,
    *,
    features: Any,
    fingerprint: str,
    max_shard_size: str | int = "1GB",
    writer_batch_size: int = 1024,
    progress_total: int | None = None,
    progress_description: str = "Writing Arrow dataset",
) -> ArrowStoreResult:
    """Write a loadable Arrow dataset in one pass without an HF cache copy."""

    from datasets.arrow_writer import ArrowWriter

    if not fingerprint:
        raise ValueError("fingerprint cannot be empty")
    if writer_batch_size <= 0:
        raise ValueError("writer_batch_size must be positive")
    maximum_bytes = _maximum_bytes(max_shard_size)
    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    columns = list(features.keys())
    column_set = set(columns)
    buffer = {column: [] for column in columns}
    files: list[str] = []
    total_records = 0
    total_bytes = 0
    writer = None

    def open_writer():
        shard_name = f"data-{len(files):05d}.arrow"
        files.append(shard_name)
        return ArrowWriter(
            features=features,
            path=str(directory / shard_name),
            fingerprint=fingerprint,
            writer_batch_size=writer_batch_size,
        )

    def finish_writer() -> None:
        nonlocal writer, total_bytes
        if writer is None:
            return
        _records, byte_count = writer.finalize()
        total_bytes += int(byte_count)
        writer = None

    def flush() -> None:
        nonlocal writer
        if not buffer[columns[0]]:
            return
        if writer is None:
            writer = open_writer()
        writer.write_batch(buffer)
        for values in buffer.values():
            values.clear()
        if int(getattr(writer, "_num_bytes", 0)) >= maximum_bytes:
            finish_writer()

    try:
        from tqdm import tqdm

        with tqdm(
            total=progress_total,
            desc=progress_description,
            unit=" docs",
            disable=None,
        ) as progress:
            for row in rows:
                missing = column_set - row.keys()
                extra = row.keys() - column_set
                if missing or extra:
                    raise ValueError(
                        f"Arrow row schema mismatch; missing={sorted(missing)}, "
                        f"extra={sorted(extra)}"
                    )
                for column in columns:
                    buffer[column].append(row[column])
                total_records += 1
                progress.update(1)
                if len(buffer[columns[0]]) >= writer_batch_size:
                    flush()
        flush()
        finish_writer()
    except BaseException:
        if writer is not None:
            writer.close()
        raise

    if total_records == 0:
        raise ValueError("cannot write an empty Arrow dataset")
    descriptor = {
        "store_version": ARROW_STORE_VERSION,
        "format": "huggingface_arrow_files",
        "fingerprint": fingerprint,
        "records": total_records,
        "bytes_written": total_bytes,
        "files": files,
        "features": features.to_dict(),
    }
    (directory / DESCRIPTOR_NAME).write_text(
        json.dumps(descriptor, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ArrowStoreResult(
        records=total_records,
        bytes_written=total_bytes,
        fingerprint=fingerprint,
        files=tuple(files),
    )


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
