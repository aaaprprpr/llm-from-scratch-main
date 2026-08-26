from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .identity import sha256_text, stable_json


SCHEMA_VERSION = "1"


class Decision(str, Enum):
    UNREVIEWED = "unreviewed"
    KEEP = "keep"
    DROP = "drop"
    UNSURE = "unsure"


class PrimaryCategory(str, Enum):
    ENCYCLOPEDIA = "encyclopedia"
    NEWS = "news"
    MARKETING = "marketing"
    FICTION = "fiction"
    FORUM_OR_SOCIAL = "forum_or_social"
    QA_OR_INSTRUCTION = "qa_or_instruction"
    ACADEMIC_OR_TECHNICAL = "academic_or_technical"
    CODE = "code"
    REFERENCE_OR_TABLE = "reference_or_table"
    OTHER = "other"


REVIEW_FLAGS = frozenset(
    {
        "advertisement",
        "boilerplate",
        "spam",
        "garbled",
        "duplicate",
        "low_information",
        "bad_format",
        "unsafe_or_pii",
        "wrong_language",
        "irrelevant",
        "other",
    }
)


@dataclass(frozen=True)
class ImportedDocument:
    stable_locator: str
    text: str
    source_local_id: str | None = None
    title: str | None = None
    url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stable_locator:
            raise ValueError("ImportedDocument.stable_locator cannot be empty")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("ImportedDocument.text must be a nonempty string")
        stable_json(dict(self.metadata))


@dataclass(frozen=True)
class FieldMapping:
    text_fields: tuple[str, ...]
    text_separator: str = "\n\n"
    title_field: str | None = None
    url_field: str | None = None
    local_id_field: str | None = None
    metadata_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.text_fields:
            raise ValueError("FieldMapping.text_fields cannot be empty")
        if not all(isinstance(value, str) and value for value in self.text_fields):
            raise ValueError("text_fields must contain nonempty strings")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["text_fields"] = list(self.text_fields)
        value["metadata_fields"] = list(self.metadata_fields)
        return value


@dataclass(frozen=True)
class Inspection:
    source_type: str
    input_location: str
    input_fingerprint: str
    fields: tuple[str, ...]
    record_count: int | None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceManifest:
    schema_version: str
    source_id: str
    source_revision: str
    source_type: str
    original_location: str
    input_fingerprint: str
    adapter_name: str
    adapter_version: str
    mapping: Mapping[str, Any]
    license: str
    imported_at: str
    record_count: int
    content_sequence_sha256: str
    dataset_path: str
    dataset_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_features():
    from datasets import Features, Value

    return Features(
        {
            "schema_version": Value("string"),
            "doc_id": Value("string"),
            "source_id": Value("string"),
            "source_revision": Value("string"),
            "source_row": Value("int64"),
            "source_local_id": Value("string"),
            "text": Value("string"),
            "title": Value("string"),
            "url": Value("string"),
            "content_sha256": Value("string"),
            "metadata_json": Value("string"),
        }
    )


def canonical_row(
    *,
    doc_id: str,
    source_id: str,
    source_revision: str,
    source_row: int,
    document: ImportedDocument,
) -> dict[str, Any]:
    if source_row < 0:
        raise ValueError("source_row must be non-negative")
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "source_id": source_id,
        "source_revision": source_revision,
        "source_row": source_row,
        "source_local_id": document.source_local_id,
        "text": document.text,
        "title": document.title,
        "url": document.url,
        "content_sha256": sha256_text(document.text),
        "metadata_json": stable_json(dict(document.metadata)),
    }


def write_manifest(path: Path, manifest: SourceManifest | Mapping[str, Any]) -> None:
    value = manifest.to_dict() if isinstance(manifest, SourceManifest) else dict(manifest)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
