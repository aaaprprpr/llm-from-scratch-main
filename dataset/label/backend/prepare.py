from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .identity import sha256_text, stable_json
from .dataset_store import load_dataset, write_arrow_dataset
from .schema import SCHEMA_VERSION, SourceManifest, canonical_features, write_manifest


PREPARE_VERSION = "normalize_v1"
_CONTROL_CHARACTER_TRANSLATION = {
    codepoint: None
    for codepoint in (*range(0x20), *range(0x7F, 0xA0))
    if codepoint not in {0x09, 0x0A}
}
_LETTER_PATTERN = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class PrepareConfig:
    """Deterministic, one-to-one normalization before human review.

    V0.1 deliberately does not drop records.  That keeps ``source_row`` equal to
    the physical Arrow row and makes every imported record inspectable.  Quality
    filtering belongs to explicit review decisions or a later, versioned rule
    stage.
    """

    normalize_newlines: bool = True
    unicode_normalization: str = "NFC"
    remove_bom: bool = True
    remove_control_characters: bool = True
    strip_text: bool = True
    fix_text: bool = False

    def __post_init__(self) -> None:
        if self.unicode_normalization not in {"NFC", "NFKC", "none"}:
            raise ValueError("unicode_normalization must be NFC, NFKC, or none")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrepareManifest:
    schema_version: str
    prepare_version: str
    prepare_revision: str
    source_id: str
    source_revision: str
    source_manifest_path: str
    input_dataset_path: str
    input_dataset_fingerprint: str | None
    config: Mapping[str, Any]
    config_hash: str
    created_at: str
    input_records: int
    output_records: int
    changed_records: int
    diagnostics: Mapping[str, int]
    content_sequence_sha256: str
    dataset_path: str
    dataset_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrepareResult:
    manifest: PrepareManifest
    revision_directory: Path
    reused_existing: bool


def normalize_review_text(text: str, config: PrepareConfig) -> str:
    if not isinstance(text, str):
        raise TypeError("review text must be a string")
    result = text
    if config.fix_text:
        import ftfy

        result = ftfy.fix_text(result)
    if config.normalize_newlines:
        result = result.replace("\r\n", "\n").replace("\r", "\n")
    if config.unicode_normalization != "none":
        result = unicodedata.normalize(config.unicode_normalization, result)
    if config.remove_bom:
        result = result.replace("\ufeff", "")
    if config.remove_control_characters:
        result = result.translate(_CONTROL_CHARACTER_TRANSLATION)
    if config.strip_text:
        result = result.strip()
    if not result:
        raise ValueError("normalization produced empty review text")
    return result


def _update_length_prefixed(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _prepare_revision(source_revision: str, config: PrepareConfig) -> tuple[str, str]:
    config_json = stable_json(config.to_dict())
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    revision_digest = hashlib.blake2b(digest_size=16)
    for value in (source_revision, PREPARE_VERSION, config_hash, SCHEMA_VERSION):
        _update_length_prefixed(revision_digest, value)
    return revision_digest.hexdigest(), config_hash


def _validate_canonical_row(row: Mapping[str, Any], source: SourceManifest) -> None:
    required = {
        "schema_version",
        "doc_id",
        "source_id",
        "source_revision",
        "source_row",
        "text",
        "content_sha256",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"canonical input row is missing fields: {sorted(missing)}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical schema: {row['schema_version']!r}")
    if row["source_id"] != source.source_id:
        raise ValueError("canonical row source_id does not match source manifest")
    if row["source_revision"] != source.source_revision:
        raise ValueError("canonical row source_revision does not match manifest")
    if sha256_text(row["text"]) != row["content_sha256"]:
        raise ValueError(f"raw content hash mismatch for doc_id {row['doc_id']}")


class PrepareService:
    def prepare_source(
        self,
        source_revision_directory: str | Path,
        *,
        config: PrepareConfig | None = None,
        max_shard_size: str | int = "1GB",
        read_batch_size: int = 1024,
    ) -> PrepareResult:
        config = config or PrepareConfig()
        if read_batch_size <= 0:
            raise ValueError("read_batch_size must be positive")
        source_directory = Path(source_revision_directory).resolve()
        source_manifest_path = source_directory / "manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(source_manifest_path)
        source = SourceManifest(
            **json.loads(source_manifest_path.read_text(encoding="utf-8"))
        )
        input_dataset_path = source_directory / source.dataset_path
        if not input_dataset_path.exists():
            raise FileNotFoundError(input_dataset_path)
        input_dataset = load_dataset(input_dataset_path)
        if set(input_dataset.column_names) != set(canonical_features().keys()):
            raise ValueError("raw source does not use the canonical dataset schema")

        prepare_revision, config_hash = _prepare_revision(
            source.source_revision,
            config,
        )
        revision_directory = source_directory / "review" / prepare_revision
        manifest_path = revision_directory / "manifest.json"
        if revision_directory.exists():
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Existing prepare revision has no manifest: {revision_directory}"
                )
            manifest = PrepareManifest(
                **json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            return PrepareResult(manifest, revision_directory, True)

        parent = revision_directory.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary_directory = parent / f".{prepare_revision}.tmp-{uuid.uuid4().hex}"
        temporary_directory.mkdir()
        output_dataset_path = temporary_directory / "dataset"
        diagnostics: Counter[str] = Counter()
        output_count = 0
        changed_count = 0
        sequence_digest = hashlib.sha256()

        def generate_rows():
            nonlocal output_count, changed_count
            for batch in input_dataset.iter(batch_size=read_batch_size):
                batch_count = len(next(iter(batch.values()), ()))
                for batch_offset in range(batch_count):
                    dataset_row = output_count
                    row = {
                        column: values[batch_offset]
                        for column, values in batch.items()
                    }
                    _validate_canonical_row(row, source)
                    if int(row["source_row"]) != dataset_row:
                        raise ValueError(
                            "V0.1 Prepare requires one-to-one source rows; "
                            f"dataset row {dataset_row} stores source_row "
                            f"{row['source_row']}"
                        )
                    text = normalize_review_text(row["text"], config)
                    if text != row["text"]:
                        changed_count += 1
                    if "\ufffd" in text:
                        diagnostics["contains_replacement_character"] += 1
                    if _LETTER_PATTERN.search(text) is None:
                        diagnostics["contains_no_letters"] += 1
                    output = dict(row)
                    output["text"] = text
                    output["content_sha256"] = sha256_text(text)
                    _update_length_prefixed(sequence_digest, output["doc_id"])
                    _update_length_prefixed(
                        sequence_digest, output["content_sha256"]
                    )
                    output_count += 1
                    yield output

        try:
            store = write_arrow_dataset(
                generate_rows(),
                output_dataset_path,
                features=canonical_features(),
                fingerprint=prepare_revision,
                max_shard_size=max_shard_size,
                progress_total=len(input_dataset),
                progress_description=f"Preparing {source.source_id}",
            )
            if store.records != len(input_dataset) or output_count != store.records:
                raise RuntimeError("Prepare changed row cardinality")
            input_fingerprint = getattr(input_dataset, "_fingerprint", None)
            manifest = PrepareManifest(
                schema_version=SCHEMA_VERSION,
                prepare_version=PREPARE_VERSION,
                prepare_revision=prepare_revision,
                source_id=source.source_id,
                source_revision=source.source_revision,
                source_manifest_path=str(source_manifest_path),
                input_dataset_path=str(input_dataset_path),
                input_dataset_fingerprint=input_fingerprint,
                config=config.to_dict(),
                config_hash=config_hash,
                created_at=datetime.now(UTC).isoformat(timespec="microseconds"),
                input_records=len(input_dataset),
                output_records=output_count,
                changed_records=changed_count,
                diagnostics=dict(sorted(diagnostics.items())),
                content_sequence_sha256=sequence_digest.hexdigest(),
                dataset_path="dataset",
                dataset_fingerprint=store.fingerprint,
            )
            write_manifest(temporary_directory / "manifest.json", manifest.to_dict())
            temporary_directory.replace(revision_directory)
            return PrepareResult(manifest, revision_directory, False)
        except BaseException:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)
            raise
