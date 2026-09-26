from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    path: Path
    dataset: Any

    @property
    def records(self) -> int:
        return len(self.dataset)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if "sampling" not in config:
        raise ValueError(f"Missing 'sampling' section in {CONFIG_PATH}")
    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def discover_dataset_directories(input_paths: list[str]) -> list[Path]:
    discovered: set[Path] = set()
    for configured_path in input_paths:
        root = resolve_path(configured_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset search path does not exist: {root}")
        if root.is_file():
            raise ValueError(
                f"Expected a directory containing a saved Hugging Face dataset: {root}"
            )

        if (root / "state.json").is_file() and (root / "dataset_info.json").is_file():
            discovered.add(root)
            continue

        for state_file in root.rglob("state.json"):
            candidate = state_file.parent
            if (candidate / "dataset_info.json").is_file():
                discovered.add(candidate.resolve())

    if not discovered:
        joined = ", ".join(input_paths)
        raise FileNotFoundError(
            "No Hugging Face save_to_disk dataset was found under: " + joined
        )
    return sorted(discovered)


def load_sources(dataset_paths: list[Path]) -> list[DatasetSource]:
    sources: list[DatasetSource] = []
    for path in dataset_paths:
        print(f"Loading dataset: {path}")
        loaded = load_from_disk(str(path), keep_in_memory=False)
        if isinstance(loaded, DatasetDict):
            for split_name, split in loaded.items():
                sources.append(
                    DatasetSource(
                        name=f"{path.name}/{split_name}", path=path, dataset=split
                    )
                )
        else:
            sources.append(DatasetSource(name=path.name, path=path, dataset=loaded))
    return sources


def validate_sources(sources: list[DatasetSource], text_fields: list[str]) -> None:
    if not text_fields:
        raise ValueError("sampling.text_fields must contain at least one field")
    for source in sources:
        available = set(source.dataset.column_names)
        if not available.intersection(text_fields):
            raise ValueError(
                f"Dataset {source.name} has none of the configured text fields "
                f"{text_fields}; available columns: {sorted(available)}"
            )


def compose_text(
    row: dict[str, Any],
    text_fields: list[str],
    field_separator: str,
    deduplicate_leading_title: bool,
) -> str:
    values = []
    for field in text_fields:
        value = row.get(field)
        if isinstance(value, str):
            value = value.strip()
            if value:
                values.append(value)

    if (
        deduplicate_leading_title
        and len(values) >= 2
        and values[1].startswith(values[0])
    ):
        values = values[1:]
    return field_separator.join(values)


def rows_from_batch(batch: dict[str, list[Any]], size: int):
    columns = tuple(batch)
    for position in range(size):
        yield {column: batch[column][position] for column in columns}


def sample_dataset(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["sampling"]
    target_size_mib = int(settings["target_size_mib"])
    if target_size_mib <= 0:
        raise ValueError("sampling.target_size_mib must be positive")
    target_bytes = target_size_mib * 1024 * 1024

    batch_records = int(settings.get("batch_records", 2048))
    if batch_records <= 0:
        raise ValueError("sampling.batch_records must be positive")

    text_fields = list(settings["text_fields"])
    dataset_paths = discover_dataset_directories(list(settings["input_paths"]))
    sources = load_sources(dataset_paths)
    validate_sources(sources, text_fields)

    output_path = resolve_path(settings["output_file"])
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    overwrite = bool(settings.get("overwrite", False))
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Set sampling.overwrite=true to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    source_ends = []
    total_records = 0
    for source in sources:
        total_records += source.records
        source_ends.append(total_records)
        print(
            f"Found dataset: {source.name} | records={source.records:,} | "
            f"columns={source.dataset.column_names}"
        )
    if total_records == 0:
        raise ValueError("The discovered datasets contain no records")

    seed = int(settings.get("seed", 42))
    rng = random.Random(seed)
    used_global_indices: set[int] = set()
    written_records = 0
    skipped_records = 0
    output_bytes = 0
    digest = hashlib.sha256()
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_field = settings.get("category_field")
    separator = str(settings.get("record_separator", "\n\n")).encode("utf-8")
    field_separator = str(settings.get("field_separator", "\n"))
    min_characters = int(settings.get("min_text_characters", 1))
    max_characters = settings.get("max_text_characters_per_record")
    if max_characters is not None:
        max_characters = int(max_characters)
        if max_characters <= 0:
            raise ValueError(
                "sampling.max_text_characters_per_record must be null or positive"
            )

    try:
        with temporary_path.open("wb") as output, tqdm(
            total=target_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Sampling tokenizer corpus",
            dynamic_ncols=True,
        ) as progress:
            while output_bytes < target_bytes and len(used_global_indices) < total_records:
                requested = min(
                    batch_records, total_records - len(used_global_indices)
                )
                selected: list[int] = []
                while len(selected) < requested:
                    global_index = rng.randrange(total_records)
                    if global_index in used_global_indices:
                        continue
                    used_global_indices.add(global_index)
                    selected.append(global_index)

                grouped: dict[int, list[int]] = defaultdict(list)
                for global_index in selected:
                    source_index = bisect.bisect_right(source_ends, global_index)
                    source_start = 0 if source_index == 0 else source_ends[source_index - 1]
                    grouped[source_index].append(global_index - source_start)

                for source_index, local_indices in grouped.items():
                    source = sources[source_index]
                    batch = source.dataset[local_indices]
                    for row in rows_from_batch(batch, len(local_indices)):
                        text = compose_text(
                            row,
                            text_fields,
                            field_separator,
                            bool(settings.get("deduplicate_leading_title", True)),
                        )
                        if len(text) < min_characters:
                            skipped_records += 1
                            continue
                        if max_characters is not None and len(text) > max_characters:
                            text = text[:max_characters]

                        encoded = text.encode("utf-8")
                        payload = encoded if written_records == 0 else separator + encoded
                        output.write(payload)
                        digest.update(payload)
                        output_bytes += len(payload)
                        written_records += 1
                        source_counts[source.name] += 1
                        if category_field:
                            category = row.get(category_field)
                            category_counts[str(category or "<missing>")] += 1
                        progress.update(len(payload))
                        if output_bytes >= target_bytes:
                            break
                    if output_bytes >= target_bytes:
                        break

        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    metadata = {
        "output_file": str(output_path),
        "target_bytes": target_bytes,
        "actual_bytes": output_bytes,
        "records_written": written_records,
        "records_skipped": skipped_records,
        "seed": seed,
        "sha256": digest.hexdigest(),
        "text_fields": text_fields,
        "dataset_paths": [str(path) for path in dataset_paths],
        "available_records": total_records,
        "sampled_records_by_source": dict(source_counts),
        "sampled_records_by_category": dict(category_counts),
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")

    print(
        f"Saved {written_records:,} records, {output_bytes / (1024 ** 2):.2f} MiB "
        f"to {output_path}"
    )
    print(f"Metadata: {metadata_path}")
    return metadata


def main() -> None:
    sample_dataset(load_config())


if __name__ == "__main__":
    main()
