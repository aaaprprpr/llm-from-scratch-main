from __future__ import annotations

import json
import hashlib
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .identity import make_doc_id, make_source_revision, validate_source_id
from .dataset_store import write_arrow_dataset
from .importers.base import SourceAdapter, SourceSpec
from .schema import (
    SCHEMA_VERSION,
    FieldMapping,
    SourceManifest,
    canonical_features,
    canonical_row,
    write_manifest,
)


@dataclass(frozen=True)
class ImportResult:
    source_manifest: SourceManifest
    revision_directory: Path
    reused_existing: bool


class ImportService:
    def __init__(self, storage_root: str | Path):
        self.storage_root = Path(storage_root).resolve()

    def import_source(
        self,
        *,
        source_id: str,
        source_license: str,
        adapter: SourceAdapter,
        spec: SourceSpec,
        mapping: FieldMapping,
        max_shard_size: str | int = "1GB",
    ) -> ImportResult:
        validate_source_id(source_id)
        inspection = adapter.inspect(spec)
        revision = make_source_revision(
            source_id=source_id,
            adapter_name=adapter.adapter_name,
            adapter_version=adapter.adapter_version,
            input_fingerprint=inspection.input_fingerprint,
            mapping=mapping.to_dict(),
            schema_version=SCHEMA_VERSION,
        )
        source_directory = self.storage_root / "sources" / source_id
        revision_directory = source_directory / revision
        manifest_path = revision_directory / "manifest.json"
        if revision_directory.exists():
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Existing source revision has no manifest: {revision_directory}"
                )
            manifest = SourceManifest(**json.loads(manifest_path.read_text("utf-8")))
            requested_license = source_license or "unknown"
            if manifest.license != requested_license:
                raise ValueError(
                    "This source revision already exists with a different license: "
                    f"stored={manifest.license!r}, requested={requested_license!r}"
                )
            return ImportResult(manifest, revision_directory, True)

        source_directory.mkdir(parents=True, exist_ok=True)
        temporary_directory = source_directory / f".{revision}.tmp-{uuid.uuid4().hex}"
        temporary_directory.mkdir()
        raw_directory = temporary_directory / "raw"
        imported_count = 0
        sequence_digest = hashlib.sha256()

        def update_digest(value: str) -> None:
            encoded = value.encode("utf-8")
            sequence_digest.update(len(encoded).to_bytes(8, "big"))
            sequence_digest.update(encoded)

        def generate_rows():
            nonlocal imported_count
            for source_row, document in enumerate(
                adapter.iter_documents(spec, mapping)
            ):
                doc_id = make_doc_id(
                    source_id,
                    revision,
                    document.stable_locator,
                )
                row = canonical_row(
                    doc_id=doc_id,
                    source_id=source_id,
                    source_revision=revision,
                    source_row=source_row,
                    document=document,
                )
                update_digest(row["doc_id"])
                update_digest(row["content_sha256"])
                yield row
                imported_count += 1

        try:
            store = write_arrow_dataset(
                generate_rows(),
                raw_directory,
                features=canonical_features(),
                fingerprint=revision,
                max_shard_size=max_shard_size,
                progress_description=f"Importing {source_id}",
            )
            if imported_count != store.records:
                raise RuntimeError(
                    f"generator counted {imported_count} rows but Arrow has "
                    f"{store.records}"
                )

            manifest = SourceManifest(
                schema_version=SCHEMA_VERSION,
                source_id=source_id,
                source_revision=revision,
                source_type=inspection.source_type,
                original_location=inspection.input_location,
                input_fingerprint=inspection.input_fingerprint,
                adapter_name=adapter.adapter_name,
                adapter_version=adapter.adapter_version,
                mapping=mapping.to_dict(),
                license=source_license or "unknown",
                imported_at=datetime.now(UTC).isoformat(timespec="microseconds"),
                record_count=imported_count,
                content_sequence_sha256=sequence_digest.hexdigest(),
                dataset_path="raw",
                dataset_fingerprint=store.fingerprint,
            )
            write_manifest(temporary_directory / "manifest.json", manifest)
            temporary_directory.replace(revision_directory)
            return ImportResult(manifest, revision_directory, False)
        except BaseException:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)
            raise
