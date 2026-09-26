from __future__ import annotations

from typing import Iterator

from ..schema import FieldMapping, ImportedDocument, Inspection
from .base import (
    SourceAdapter,
    SourceSpec,
    file_inspection_fingerprint,
    mapped_document,
)


class TextAdapter(SourceAdapter):
    adapter_name = "text"
    adapter_version = "1"

    def inspect(self, spec: SourceSpec) -> Inspection:
        path = spec.path
        strategy = str(spec.options.get("document_strategy", "blank_line_separator"))
        return Inspection(
            source_type="text",
            input_location=str(path),
            input_fingerprint=file_inspection_fingerprint(path),
            fields=("text",),
            record_count=None,
            details={
                "encoding": str(spec.options.get("encoding", "utf-8")),
                "document_strategy": strategy,
                "size_bytes": path.stat().st_size,
            },
        )

    def iter_documents(
        self,
        spec: SourceSpec,
        mapping: FieldMapping,
    ) -> Iterator[ImportedDocument]:
        strategy = str(spec.options.get("document_strategy", "blank_line_separator"))
        encoding = str(spec.options.get("encoding", "utf-8"))
        if strategy == "whole_file":
            maximum = int(spec.options.get("max_whole_file_bytes", 64 * 1024 * 1024))
            size = spec.path.stat().st_size
            if size > maximum:
                raise ValueError(
                    f"whole_file input is {size} bytes, above configured limit {maximum}"
                )
            text = spec.path.read_text(encoding=encoding)
            if text.strip():
                yield mapped_document(
                    {"text": text},
                    mapping,
                    stable_locator="whole_file:0",
                )
            return

        if strategy == "one_line_per_document":
            yield from self._iter_lines(spec, mapping, encoding)
            return

        if strategy not in {"blank_line_separator", "custom_separator"}:
            raise ValueError(f"Unsupported TXT document_strategy: {strategy}")
        custom_separator = spec.options.get("custom_separator")
        if strategy == "custom_separator" and not isinstance(custom_separator, str):
            raise ValueError("custom_separator strategy requires a string separator")
        yield from self._iter_groups(
            spec,
            mapping,
            encoding,
            custom_separator=custom_separator if strategy == "custom_separator" else None,
        )

    @staticmethod
    def _iter_lines(
        spec: SourceSpec,
        mapping: FieldMapping,
        encoding: str,
    ) -> Iterator[ImportedDocument]:
        with spec.path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                line_number += 1
                text = raw_line.decode(encoding).rstrip("\r\n")
                if not text.strip():
                    continue
                yield mapped_document(
                    {"text": text},
                    mapping,
                    stable_locator=f"line:{line_number}:byte:{offset}",
                )

    @staticmethod
    def _iter_groups(
        spec: SourceSpec,
        mapping: FieldMapping,
        encoding: str,
        *,
        custom_separator: str | None,
    ) -> Iterator[ImportedDocument]:
        with spec.path.open("rb") as stream:
            lines: list[str] = []
            document_offset = 0
            document_number = 0

            def flush() -> ImportedDocument | None:
                nonlocal lines, document_number
                text = "".join(lines).strip("\r\n")
                lines = []
                if not text.strip():
                    return None
                document_number += 1
                return mapped_document(
                    {"text": text},
                    mapping,
                    stable_locator=(
                        f"document:{document_number}:byte:{document_offset}"
                    ),
                )

            while True:
                offset = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                line = raw_line.decode(encoding)
                line_without_newline = line.rstrip("\r\n")
                is_separator = (
                    not line_without_newline.strip()
                    if custom_separator is None
                    else line_without_newline == custom_separator
                )
                if is_separator:
                    document = flush()
                    if document is not None:
                        yield document
                    document_offset = stream.tell()
                    continue
                if not lines:
                    document_offset = offset
                # Normalize platform line endings.  Stable source locations still
                # use byte offsets, while canonical text is identical on Linux and
                # Windows.
                lines.append(line_without_newline + "\n")

            document = flush()
            if document is not None:
                yield document
