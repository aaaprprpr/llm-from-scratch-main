from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..schema import FieldMapping, ImportedDocument, Inspection


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())


class SourceAdapter(ABC):
    adapter_name: str
    adapter_version: str

    @abstractmethod
    def inspect(self, spec: SourceSpec) -> Inspection:
        raise NotImplementedError

    def preview(
        self,
        spec: SourceSpec,
        mapping: FieldMapping,
        limit: int = 20,
    ) -> list[ImportedDocument]:
        if limit <= 0:
            raise ValueError("preview limit must be positive")
        documents = []
        for document in self.iter_documents(spec, mapping):
            documents.append(document)
            if len(documents) == limit:
                break
        return documents

    @abstractmethod
    def iter_documents(
        self,
        spec: SourceSpec,
        mapping: FieldMapping,
    ) -> Iterator[ImportedDocument]:
        raise NotImplementedError


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inspection_fingerprint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def get_nested_value(record: Mapping[str, Any], field_path: str) -> Any:
    value: Any = record
    for component in field_path.split("."):
        if isinstance(value, Mapping):
            if component not in value:
                return None
            value = value[component]
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            try:
                value = value[int(component)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return value


def json_compatible(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [json_compatible(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return json_compatible(item_method())
    return str(value)


def mapped_document(
    record: Mapping[str, Any],
    mapping: FieldMapping,
    *,
    stable_locator: str,
) -> ImportedDocument:
    text_parts = []
    if mapping.record_adapter:
        from data_pipeline.record_adapters import ADAPTERS
        adapted = ADAPTERS[mapping.record_adapter](record)
        if adapted:
            text_parts.append(adapted)
    else:
        for field_path in mapping.text_fields:
            value = get_nested_value(record, field_path)
            if isinstance(value, str) and value.strip():
                text_parts.append(value)
            elif value is not None and not isinstance(value, str):
                raise ValueError(f"field {field_path} is structured data; select a record adapter")
    if not text_parts:
        raise ValueError(
            f"record {stable_locator} has no nonempty mapped text fields "
            f"{mapping.text_fields}"
        )

    def optional_string(field_path: str | None) -> str | None:
        if field_path is None:
            return None
        value = get_nested_value(record, field_path)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    metadata = {
        field_path: json_compatible(get_nested_value(record, field_path))
        for field_path in mapping.metadata_fields
    }
    return ImportedDocument(
        stable_locator=stable_locator,
        text=mapping.text_separator.join(text_parts),
        source_local_id=optional_string(mapping.local_id_field),
        title=optional_string(mapping.title_field),
        url=optional_string(mapping.url_field),
        metadata=metadata,
    )


def flatten_field_names(
    value: Any,
    *,
    prefix: str = "",
    maximum_depth: int = 4,
) -> set[str]:
    fields: set[str] = set()
    if maximum_depth < 0:
        return fields
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            fields.add(path)
            fields.update(
                flatten_field_names(
                    child,
                    prefix=path,
                    maximum_depth=maximum_depth - 1,
                )
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ) and value:
        path = f"{prefix}.0" if prefix else "0"
        fields.add(path)
        fields.update(
            flatten_field_names(
                value[0],
                prefix=path,
                maximum_depth=maximum_depth - 1,
            )
        )
    return fields
