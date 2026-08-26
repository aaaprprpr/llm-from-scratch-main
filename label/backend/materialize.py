from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .blocks import BLOCK_PARSER_VERSION, parse_blocks, render_blocks
from .database import CurationDatabase
from .dataset_store import write_arrow_dataset
from .identity import sha256_text, stable_json
from .queueing import OpenProjectSource, open_project_sources
from .schema import Decision, SCHEMA_VERSION, canonical_features, write_manifest


MATERIALIZE_VERSION = "1"
POLICIES = frozenset({"keep_only", "drop_rejected"})


@dataclass(frozen=True)
class MaterializationManifest:
    schema_version: str
    materialize_version: str
    materialization_id: str
    project_id: str
    snapshot_event_seq: int
    policy: str
    block_parser_version: str
    config_hash: str
    created_at: str
    sources: tuple[Mapping[str, Any], ...]
    input_records: int
    output_records: int
    decision_counts: Mapping[str, int]
    source_input_counts: Mapping[str, int]
    source_output_counts: Mapping[str, int]
    category_counts: Mapping[str, int]
    flag_counts: Mapping[str, int]
    block_drop_count: int
    content_sequence_sha256: str
    dataset_path: str
    dataset_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaterializationResult:
    manifest: MaterializationManifest
    output_directory: Path
    reused_existing: bool


def _digest_value(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _configuration(
    *,
    project_id: str,
    snapshot_event_seq: int,
    policy: str,
    sources: list[OpenProjectSource],
) -> tuple[str, str]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "materialize_version": MATERIALIZE_VERSION,
        "project_id": project_id,
        "snapshot_event_seq": snapshot_event_seq,
        "policy": policy,
        "block_parser_version": BLOCK_PARSER_VERSION,
        "sources": [
            {
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "row_count": source.row_count,
                "dataset_fingerprint": getattr(source.dataset, "_fingerprint", None),
            }
            for source in sources
        ],
    }
    serialized = stable_json(value)
    config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    materialization_id = hashlib.blake2b(
        serialized.encode("utf-8"), digest_size=16
    ).hexdigest()
    return materialization_id, config_hash


def _manifest_from_path(path: Path) -> MaterializationManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["sources"] = tuple(value["sources"])
    return MaterializationManifest(**value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MaterializeService:
    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root).resolve()

    def materialize(
        self,
        database: CurationDatabase,
        *,
        project_id: str,
        policy: str,
        snapshot_event_seq: int | None = None,
        max_shard_size: str | int = "1GB",
        read_batch_size: int = 1024,
    ) -> MaterializationResult:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {sorted(POLICIES)}")
        if read_batch_size <= 0:
            raise ValueError("read_batch_size must be positive")
        latest_event_seq = database.latest_event_seq(project_id)
        if snapshot_event_seq is None:
            snapshot_event_seq = latest_event_seq
        if not 0 <= snapshot_event_seq <= latest_event_seq:
            raise ValueError(
                f"snapshot_event_seq must be between 0 and {latest_event_seq}"
            )

        sources = open_project_sources(database, project_id)
        materialization_id, config_hash = _configuration(
            project_id=project_id,
            snapshot_event_seq=snapshot_event_seq,
            policy=policy,
            sources=sources,
        )
        output_directory = self.output_root / materialization_id
        manifest_path = output_directory / "manifest.json"
        if output_directory.exists():
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Existing materialization has no manifest: {output_directory}"
                )
            manifest = _manifest_from_path(manifest_path)
            manifest_sha256 = _file_sha256(manifest_path)
            try:
                record = database.get_materialization(materialization_id)
            except KeyError:
                database.record_materialization(
                    materialization_id=materialization_id,
                    project_id=project_id,
                    snapshot_event_seq=snapshot_event_seq,
                    policy={"type": policy},
                    config_hash=config_hash,
                    output_path=output_directory,
                    manifest_sha256=manifest_sha256,
                )
            else:
                if record["manifest_sha256"] != manifest_sha256:
                    raise RuntimeError("materialization manifest changed after creation")
            return MaterializationResult(manifest, output_directory, True)

        document_states, block_states = database.review_state_at(
            project_id,
            snapshot_event_seq,
        )
        blocks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for state in block_states.values():
            blocks_by_document[state["doc_id"]].append(state)

        self.output_root.mkdir(parents=True, exist_ok=True)
        temporary_directory = self.output_root / (
            f".{materialization_id}.tmp-{uuid.uuid4().hex}"
        )
        temporary_directory.mkdir()
        dataset_path = temporary_directory / "dataset"

        decision_counts: Counter[str] = Counter()
        source_input_counts: Counter[str] = Counter()
        source_output_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        flag_counts: Counter[str] = Counter()
        seen_document_reviews: set[str] = set()
        seen_block_reviews: set[str] = set()
        output_count = 0
        input_count = 0
        block_drop_count = 0
        sequence_digest = hashlib.sha256()

        def generate_rows():
            nonlocal input_count, output_count, block_drop_count
            for source in sources:
                source_dataset_row = 0
                for batch in source.dataset.iter(batch_size=read_batch_size):
                    batch_count = len(next(iter(batch.values()), ()))
                    for batch_offset in range(batch_count):
                        dataset_row = source_dataset_row
                        source_dataset_row += 1
                        row = {
                            column: values[batch_offset]
                            for column, values in batch.items()
                        }
                        yield from process_row(source, dataset_row, row)

            stale_documents = set(document_states) - seen_document_reviews
            stale_blocks = set(block_states) - seen_block_reviews
            if stale_documents:
                raise ValueError(
                    f"reviews reference unattached documents: {sorted(stale_documents)[:5]}"
                )
            if stale_blocks:
                raise ValueError(
                    f"reviews reference unattached blocks: {sorted(stale_blocks)[:5]}"
                )

        def process_row(
            source: OpenProjectSource,
            dataset_row: int,
            row: dict[str, Any],
        ):
            nonlocal input_count, output_count, block_drop_count
            input_count += 1
            source_input_counts[source.key] += 1
            if row["source_id"] != source.source_id:
                raise ValueError("materialize source_id mismatch")
            if row["source_revision"] != source.source_revision:
                raise ValueError("materialize source_revision mismatch")
            if int(row["source_row"]) != dataset_row:
                raise ValueError("materialize source_row mismatch")
            if sha256_text(row["text"]) != row["content_sha256"]:
                raise ValueError(
                    f"review content hash mismatch for {row['doc_id']}"
                )

            review = document_states.get(row["doc_id"])
            decision = Decision.UNREVIEWED.value
            if review is not None:
                seen_document_reviews.add(row["doc_id"])
                if int(review["source_row"]) != dataset_row:
                    raise ValueError(
                        f"review source_row mismatch for {row['doc_id']}"
                    )
                if review["content_sha256"] != row["content_sha256"]:
                    raise ValueError(f"stale document review for {row['doc_id']}")
                decision = review["decision"]
                if review["primary_category"] is not None:
                    category_counts[review["primary_category"]] += 1
                flag_counts.update(review["flags"])
            decision_counts[decision] += 1

            parsed_blocks = None
            dropped_blocks = set()
            for block_review in blocks_by_document.get(row["doc_id"], ()):
                if parsed_blocks is None:
                    parsed_blocks = parse_blocks(row["text"], row["doc_id"])
                by_id = {block.block_id: block for block in parsed_blocks}
                block = by_id.get(block_review["block_id"])
                if block is None:
                    raise ValueError(f"stale block review {block_review['block_id']}")
                if block.content_sha256 != block_review["base_block_hash"]:
                    raise ValueError(
                        f"block hash mismatch for {block_review['block_id']}"
                    )
                seen_block_reviews.add(block_review["block_id"])
                if block_review["decision"] == Decision.DROP.value:
                    dropped_blocks.add(block_review["block_id"])

            include = (
                decision == Decision.KEEP.value
                if policy == "keep_only"
                else decision != Decision.DROP.value
            )
            if not include:
                return
            text = row["text"]
            if parsed_blocks is not None:
                text = render_blocks(text, parsed_blocks, dropped_blocks)
                block_drop_count += len(dropped_blocks)
            if not text:
                raise ValueError(
                    f"materialization produced empty text for {row['doc_id']}"
                )
            content_hash = sha256_text(text)
            output = dict(row)
            output["text"] = text
            output["content_sha256"] = content_hash
            _digest_value(sequence_digest, output["doc_id"])
            _digest_value(sequence_digest, content_hash)
            output_count += 1
            source_output_counts[source.key] += 1
            yield output

        try:
            store = write_arrow_dataset(
                generate_rows(),
                dataset_path,
                features=canonical_features(),
                fingerprint=materialization_id,
                max_shard_size=max_shard_size,
                progress_description=f"Materializing {project_id}",
            )
            if output_count != store.records:
                raise RuntimeError("materialization generator count mismatch")

            source_manifests = {
                (value["source_id"], value["source_revision"]): value
                for value in database.list_project_sources(project_id)
            }
            source_values = tuple(
                {
                    "source_id": source.source_id,
                    "source_revision": source.source_revision,
                    "row_count": source.row_count,
                    "dataset_path": str(source.dataset_path),
                    "manifest_path": source_manifests[
                        (source.source_id, source.source_revision)
                    ]["manifest_path"],
                    "dataset_fingerprint": getattr(
                        source.dataset, "_fingerprint", None
                    ),
                }
                for source in sources
            )
            manifest = MaterializationManifest(
                schema_version=SCHEMA_VERSION,
                materialize_version=MATERIALIZE_VERSION,
                materialization_id=materialization_id,
                project_id=project_id,
                snapshot_event_seq=snapshot_event_seq,
                policy=policy,
                block_parser_version=BLOCK_PARSER_VERSION,
                config_hash=config_hash,
                created_at=datetime.now(UTC).isoformat(timespec="microseconds"),
                sources=source_values,
                input_records=input_count,
                output_records=output_count,
                decision_counts=dict(sorted(decision_counts.items())),
                source_input_counts=dict(sorted(source_input_counts.items())),
                source_output_counts=dict(sorted(source_output_counts.items())),
                category_counts=dict(sorted(category_counts.items())),
                flag_counts=dict(sorted(flag_counts.items())),
                block_drop_count=block_drop_count,
                content_sequence_sha256=sequence_digest.hexdigest(),
                dataset_path="dataset",
                dataset_fingerprint=store.fingerprint,
            )
            write_manifest(temporary_directory / "manifest.json", manifest.to_dict())
            temporary_directory.replace(output_directory)
            final_manifest_path = output_directory / "manifest.json"
            manifest_sha256 = _file_sha256(final_manifest_path)
            try:
                database.record_materialization(
                    materialization_id=materialization_id,
                    project_id=project_id,
                    snapshot_event_seq=snapshot_event_seq,
                    policy={"type": policy},
                    config_hash=config_hash,
                    output_path=output_directory,
                    manifest_sha256=manifest_sha256,
                )
            except BaseException:
                if output_directory.exists():
                    shutil.rmtree(output_directory)
                raise
            return MaterializationResult(manifest, output_directory, False)
        except BaseException:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)
            raise
