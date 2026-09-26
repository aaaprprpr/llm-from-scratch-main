from __future__ import annotations

from typing import Any, Iterator, Mapping

from ..identity import stable_json
from ..dataset_store import load_dataset
from ..schema import FieldMapping, ImportedDocument, Inspection
from .base import SourceAdapter, SourceSpec, flatten_field_names, mapped_document


def _iter_splits(dataset: Any):
    if isinstance(dataset, Mapping):
        for name in sorted(dataset):
            yield str(name), dataset[name]
    else:
        yield "selected", dataset


class HuggingFaceLocalAdapter(SourceAdapter):
    adapter_name = "huggingface_local"
    adapter_version = "1"

    def inspect(self, spec: SourceSpec) -> Inspection:
        dataset = load_dataset(spec.path)
        fields: set[str] = set()
        split_details = {}
        record_count = 0
        for split_name, split in _iter_splits(dataset):
            record_count += len(split)
            fields.update(split.column_names)
            if len(split):
                fields.update(flatten_field_names(split[0]))
            split_details[split_name] = {
                "rows": len(split),
                "fingerprint": getattr(split, "_fingerprint", None),
                "features": str(split.features),
            }
        fingerprint = stable_json(
            {
                "splits": split_details,
            }
        )
        return Inspection(
            source_type="huggingface_local",
            input_location=str(spec.path),
            input_fingerprint=fingerprint,
            fields=tuple(sorted(fields)),
            record_count=record_count,
            details={"splits": split_details},
        )

    def iter_documents(
        self,
        spec: SourceSpec,
        mapping: FieldMapping,
    ) -> Iterator[ImportedDocument]:
        dataset = load_dataset(spec.path)
        batch_size = int(spec.options.get("read_batch_size", 1024))
        if batch_size <= 0:
            raise ValueError("read_batch_size must be positive")
        policy = str(spec.options.get("invalid_record_policy", "error"))
        if policy not in {"error", "skip"}:
            raise ValueError("invalid_record_policy must be 'error' or 'skip'")

        for split_name, split in _iter_splits(dataset):
            for batch_start in range(0, len(split), batch_size):
                batch = split[batch_start : batch_start + batch_size]
                row_count = len(next(iter(batch.values()), []))
                for offset in range(row_count):
                    row_index = batch_start + offset
                    row = {name: values[offset] for name, values in batch.items()}
                    locator = f"split:{split_name}:row:{row_index}"
                    try:
                        yield mapped_document(
                            row,
                            mapping,
                            stable_locator=locator,
                        )
                    except ValueError as exc:
                        if policy == "skip":
                            continue
                        raise ValueError(
                            f"Invalid HF record at {locator}: {exc}"
                        ) from exc
