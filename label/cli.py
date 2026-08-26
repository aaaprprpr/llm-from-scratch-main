from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .backend.database import (
    BlockReviewInput,
    CurationDatabase,
    DocumentReviewInput,
)
from .backend.documents import DocumentService
from .backend.import_service import ImportService
from .backend.importers import ADAPTERS, SourceSpec, get_adapter
from .backend.materialize import MATERIALIZE_VERSION, MaterializeService
from .backend.prepare import PrepareConfig, PrepareService
from .backend.queueing import create_source_quota_queue, create_uniform_random_queue
from .backend.schema import FieldMapping


def _read_json(path: str | Path | None, default: Any) -> Any:
    if path is None:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mapping(path: str | Path) -> FieldMapping:
    value = _read_json(path, {})
    value["text_fields"] = tuple(value["text_fields"])
    value["metadata_fields"] = tuple(value.get("metadata_fields", ()))
    return FieldMapping(**value)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))


def _source_spec(args: argparse.Namespace) -> SourceSpec:
    return SourceSpec(Path(args.path), _read_json(args.options, {}))


def command_inspect(args: argparse.Namespace) -> None:
    inspection = get_adapter(args.adapter).inspect(_source_spec(args))
    _print_json(inspection)


def command_preview(args: argparse.Namespace) -> None:
    documents = get_adapter(args.adapter).preview(
        _source_spec(args),
        _mapping(args.mapping),
        limit=args.limit,
    )
    _print_json(documents)


def command_import(args: argparse.Namespace) -> None:
    result = ImportService(args.storage).import_source(
        source_id=args.source_id,
        source_license=args.license,
        adapter=get_adapter(args.adapter),
        spec=_source_spec(args),
        mapping=_mapping(args.mapping),
        max_shard_size=args.max_shard_size,
    )
    _print_json(result)


def command_prepare(args: argparse.Namespace) -> None:
    config = PrepareConfig(**_read_json(args.config, {}))
    result = PrepareService().prepare_source(
        args.source_revision_directory,
        config=config,
        max_shard_size=args.max_shard_size,
        read_batch_size=args.read_batch_size,
    )
    _print_json(result)


def command_project_create(args: argparse.Namespace) -> None:
    with CurationDatabase(args.database) as database:
        project_id = database.create_project(
            args.name,
            guideline_version=args.guideline_version,
            project_id=args.project_id,
        )
        _print_json(database.get_project(project_id))


def command_source_attach(args: argparse.Namespace) -> None:
    revision_directory = Path(args.prepare_revision_directory).resolve()
    manifest_path = revision_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with CurationDatabase(args.database) as database:
        database.attach_source(
            project_id=args.project_id,
            source_id=manifest["source_id"],
            source_revision=manifest["source_revision"],
            dataset_path=revision_directory / manifest["dataset_path"],
            manifest_path=manifest_path,
            row_count=int(manifest["output_records"]),
        )
        _print_json(database.list_project_sources(args.project_id))


def command_queue_uniform(args: argparse.Namespace) -> None:
    with CurationDatabase(args.database) as database:
        queue_id = create_uniform_random_queue(
            database,
            project_id=args.project_id,
            name=args.name,
            sample_size=args.sample_size,
            seed=args.seed,
            queue_id=args.queue_id,
        )
        _print_json(database.get_queue(queue_id))


def command_queue_quota(args: argparse.Namespace) -> None:
    quotas = _read_json(args.quotas, {})
    with CurationDatabase(args.database) as database:
        queue_id = create_source_quota_queue(
            database,
            project_id=args.project_id,
            name=args.name,
            quotas=quotas,
            seed=args.seed,
            queue_id=args.queue_id,
        )
        _print_json(database.get_queue(queue_id))


def command_show_item(args: argparse.Namespace) -> None:
    with CurationDatabase(args.database) as database:
        _print_json(DocumentService(database).queue_document(args.queue_id, args.ordinal))


def command_review_document(args: argparse.Namespace) -> None:
    with CurationDatabase(args.database) as database:
        document = DocumentService(database).queue_document(args.queue_id, args.ordinal)
        item = document["item"]
        row = document["document"]
        current = document["document_review"]
        expected_revision = 0 if current is None else int(current["revision"])
        state, event_seq = database.set_document_review(
            project_id=item["project_id"],
            review=DocumentReviewInput(
                doc_id=row["doc_id"],
                source_row=int(row["source_row"]),
                content_sha256=row["content_sha256"],
                decision=args.decision,
                quality=args.quality,
                primary_category=args.category,
                flags=tuple(args.flag),
                notes=args.notes,
                guideline_version=database.get_project(item["project_id"])[
                    "guideline_version"
                ],
            ),
            expected_revision=expected_revision,
            actor=args.actor,
        )
        _print_json({"review": state, "event_seq": event_seq})


def command_review_block(args: argparse.Namespace) -> None:
    with CurationDatabase(args.database) as database:
        document = DocumentService(database).queue_document(args.queue_id, args.ordinal)
        try:
            block = document["blocks"][args.block_ordinal]
        except IndexError as exc:
            raise IndexError(
                f"block ordinal {args.block_ordinal} is outside this document"
            ) from exc
        current = block["review"]
        expected_revision = 0 if current is None else int(current["revision"])
        state, event_seq = database.set_block_review(
            project_id=document["item"]["project_id"],
            review=BlockReviewInput(
                doc_id=block["doc_id"],
                block_id=block["block_id"],
                base_block_hash=block["content_sha256"],
                decision=args.decision,
                reason=args.reason,
            ),
            expected_revision=expected_revision,
            actor=args.actor,
        )
        _print_json({"review": state, "event_seq": event_seq})


def command_undo(args: argparse.Namespace) -> None:
    with CurationDatabase(args.database) as database:
        event_seq = database.undo_event(
            project_id=args.project_id,
            event_seq=args.event_seq,
            actor=args.actor,
        )
        _print_json({"undo_event_seq": event_seq})


def command_materialize(args: argparse.Namespace) -> None:
    with CurationDatabase(args.database) as database:
        result = MaterializeService(args.output_root).materialize(
            database,
            project_id=args.project_id,
            policy=args.policy,
            snapshot_event_seq=args.snapshot_event_seq,
            max_shard_size=args.max_shard_size,
            read_batch_size=args.read_batch_size,
        )
        _print_json(result)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--path", required=True)
    parser.add_argument("--options", help="JSON file containing adapter options")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic local LLM dataset curation core"
    )
    parser.add_argument("--version", action="version", version=MATERIALIZE_VERSION)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect")
    _add_source_arguments(inspect_parser)
    inspect_parser.set_defaults(func=command_inspect)

    preview = commands.add_parser("preview")
    _add_source_arguments(preview)
    preview.add_argument("--mapping", required=True)
    preview.add_argument("--limit", type=int, default=20)
    preview.set_defaults(func=command_preview)

    import_parser = commands.add_parser("import")
    _add_source_arguments(import_parser)
    import_parser.add_argument("--mapping", required=True)
    import_parser.add_argument("--storage", default="label/data")
    import_parser.add_argument("--source-id", required=True)
    import_parser.add_argument("--license", default="unknown")
    import_parser.add_argument("--max-shard-size", default="1GB")
    import_parser.set_defaults(func=command_import)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-revision-directory", required=True)
    prepare.add_argument("--config", help="JSON file containing PrepareConfig")
    prepare.add_argument("--max-shard-size", default="1GB")
    prepare.add_argument("--read-batch-size", type=int, default=1024)
    prepare.set_defaults(func=command_prepare)

    project = commands.add_parser("project-create")
    project.add_argument("--database", required=True)
    project.add_argument("--name", required=True)
    project.add_argument("--guideline-version", default="1")
    project.add_argument("--project-id")
    project.set_defaults(func=command_project_create)

    attach = commands.add_parser("source-attach")
    attach.add_argument("--database", required=True)
    attach.add_argument("--project-id", required=True)
    attach.add_argument("--prepare-revision-directory", required=True)
    attach.set_defaults(func=command_source_attach)

    uniform = commands.add_parser("queue-uniform")
    uniform.add_argument("--database", required=True)
    uniform.add_argument("--project-id", required=True)
    uniform.add_argument("--name", required=True)
    uniform.add_argument("--sample-size", type=int, required=True)
    uniform.add_argument("--seed", type=int, required=True)
    uniform.add_argument("--queue-id")
    uniform.set_defaults(func=command_queue_uniform)

    quota = commands.add_parser("queue-quota")
    quota.add_argument("--database", required=True)
    quota.add_argument("--project-id", required=True)
    quota.add_argument("--name", required=True)
    quota.add_argument("--quotas", required=True, help="JSON object file")
    quota.add_argument("--seed", type=int, required=True)
    quota.add_argument("--queue-id")
    quota.set_defaults(func=command_queue_quota)

    show = commands.add_parser("show-item")
    show.add_argument("--database", required=True)
    show.add_argument("--queue-id", required=True)
    show.add_argument("--ordinal", type=int, required=True)
    show.set_defaults(func=command_show_item)

    review_document = commands.add_parser("review-document")
    review_document.add_argument("--database", required=True)
    review_document.add_argument("--queue-id", required=True)
    review_document.add_argument("--ordinal", type=int, required=True)
    review_document.add_argument(
        "--decision", required=True, choices=("keep", "drop", "unsure")
    )
    review_document.add_argument("--quality", type=int, choices=range(4))
    review_document.add_argument("--category")
    review_document.add_argument("--flag", action="append", default=[])
    review_document.add_argument("--notes", default="")
    review_document.add_argument("--actor", default="local-cli")
    review_document.set_defaults(func=command_review_document)

    review_block = commands.add_parser("review-block")
    review_block.add_argument("--database", required=True)
    review_block.add_argument("--queue-id", required=True)
    review_block.add_argument("--ordinal", type=int, required=True)
    review_block.add_argument("--block-ordinal", type=int, required=True)
    review_block.add_argument("--decision", choices=("keep", "drop"), required=True)
    review_block.add_argument("--reason")
    review_block.add_argument("--actor", default="local-cli")
    review_block.set_defaults(func=command_review_block)

    undo = commands.add_parser("undo")
    undo.add_argument("--database", required=True)
    undo.add_argument("--project-id", required=True)
    undo.add_argument("--event-seq", type=int, required=True)
    undo.add_argument("--actor", default="local-cli")
    undo.set_defaults(func=command_undo)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--database", required=True)
    materialize.add_argument("--project-id", required=True)
    materialize.add_argument(
        "--policy", choices=("keep_only", "drop_rejected"), required=True
    )
    materialize.add_argument("--snapshot-event-seq", type=int)
    materialize.add_argument("--output-root", default="label/data/exports")
    materialize.add_argument("--max-shard-size", default="1GB")
    materialize.add_argument("--read-batch-size", type=int, default=1024)
    materialize.set_defaults(func=command_materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
