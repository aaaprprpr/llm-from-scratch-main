from __future__ import annotations

import json
import hashlib
from typing import Iterator

from ..schema import FieldMapping, ImportedDocument, Inspection
from .base import (
    SourceAdapter,
    SourceSpec,
    flatten_field_names,
    mapped_document,
    invalid_record_policy,
)


class JsonlAdapter(SourceAdapter):
    adapter_name = "jsonl"
    adapter_version = "1"

    @staticmethod
    def _parse_object(raw_line: bytes, encoding: str) -> dict:
        value = json.loads(raw_line.decode(encoding))
        if not isinstance(value, dict):
            raise ValueError("JSON value is not an object")
        return value

    def inspect(self, spec: SourceSpec) -> Inspection:
        path = spec.path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        fields: set[str] = set()
        records = 0
        invalid_records = 0
        preview_records = int(spec.options.get("inspect_records", 100))
        encoding = str(spec.options.get("encoding", "utf-8"))
        policy = invalid_record_policy(spec)
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    value = self._parse_object(raw_line, encoding)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    invalid_records += 1
                    if policy == "error":
                        raise ValueError(
                            f"Invalid JSONL record at line:{line_number}: {exc}"
                        ) from exc
                    continue
                records += 1
                if records <= preview_records:
                    fields.update(flatten_field_names(value))
        return Inspection(
            source_type="jsonl",
            input_location=str(path),
            input_fingerprint=digest.hexdigest(),
            fields=tuple(sorted(fields)),
            record_count=records,
            details={"encoding": encoding, "invalid_records": invalid_records},
        )

    def iter_documents(
        self,
        spec: SourceSpec,
        mapping: FieldMapping,
    ) -> Iterator[ImportedDocument]:
        policy = invalid_record_policy(spec)
        encoding = str(spec.options.get("encoding", "utf-8"))
        with spec.path.open("rb") as stream:
            line_number = 0
            while True:
                byte_offset = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    continue
                locator = f"line:{line_number}:byte:{byte_offset}"
                try:
                    value = self._parse_object(raw_line, encoding)
                    yield mapped_document(
                        value,
                        mapping,
                        stable_locator=locator,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    if policy == "skip":
                        continue
                    raise ValueError(f"Invalid JSONL record at {locator}: {exc}") from exc
