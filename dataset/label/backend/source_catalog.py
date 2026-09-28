"""Configured download sources and automatic field mappings for the curation UI."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from dataset.data_pipeline.config import CONFIG_PATH, PROJECT_ROOT

from .schema import FieldMapping

SAMPLE_ROOT = PROJECT_ROOT / "dataset" / "data_pipeline" / "dataset_samples"


def infer_mapping(fields: list[str]) -> FieldMapping:
    """Infer text and provenance fields from a record's shape."""
    by_lower = {name.lower(): name for name in fields}

    def find(*candidates: str) -> str | None:
        return next((by_lower[name.lower()] for name in candidates if name.lower() in by_lower), None)

    names = set(fields)
    title = find("title", "标题", "name")
    text = find("text")
    content = find("content", "body", "article", "document")
    adapter = None
    if {"楼主内容", "回复列表", "replies"} & names:
        adapter = "tieba_thread"
    elif {"messages", "conversation"} & names:
        adapter = "openai_role_content_conversation"
    elif "conversations" in names:
        adapter = "sharegpt_conversations"
    elif "instruction" in names:
        adapter = "instruction_input_output"
    elif {"question", "query", "prompt"} & names and {"answer", "output", "response"} & names:
        adapter = ("question_answer_optional_think" if {"think", "reasoning"} & names
                   else "instruction_input_output" if {"input", "context"} & names
                   else "question_answer")
    text_fields = ((text,) if text else (title, content) if title and content
                   else (content,) if content else (fields[0],) if len(fields) == 1 else ())
    metadata = find("dataType", "category", "source")
    return FieldMapping(
        text_fields=text_fields, record_adapter=adapter,
        title_field=title, url_field=find("url", "link", "source_url"),
        local_id_field=find("uniqueKey", "doc_id", "document_id", "id"),
        metadata_fields=(metadata,) if metadata else (),
    )


def _sample_rows(source_id: str, path: Path, kind: str, sample_root: Path) -> tuple[list[dict], int | None]:
    sample_path = sample_root / f"{source_id}.sample.json"
    if sample_path.is_file():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        splits = sample.get("sample_rows", {})
        rows = next(iter(splits.values()), [])
        counts = sample.get("summary", {}).get("splits", {})
        record_count = sum(value.get("num_rows", 0) for value in counts.values()) or None
        return rows[:3], record_count
    if kind == "jsonl":
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
                    if len(rows) == 3:
                        break
        return rows, None
    from .dataset_store import load_dataset

    dataset = load_dataset(path)
    if isinstance(dataset, Mapping):
        splits = list(dataset.values())
        return ([splits[0][index] for index in range(min(3, len(splits[0])))] if splits else [],
                sum(len(split) for split in splits))
    return [dataset[index] for index in range(min(3, len(dataset)))], len(dataset)


def list_download_sources(
    config_path: Path = CONFIG_PATH,
    project_root: Path = PROJECT_ROOT,
    sample_root: Path = SAMPLE_ROOT,
    imported_locations: Iterable[str | Path] = (),
) -> list[dict[str, Any]]:
    imported_paths = {Path(location).resolve() for location in imported_locations}
    sources = json.loads(config_path.read_text(encoding="utf-8"))["sources"]
    catalog = []
    for source_id, source in sources.items():
        raw_path = Path(source["path"])
        path = (raw_path if raw_path.is_absolute() else project_root / raw_path).resolve()
        if path in imported_paths:
            continue
        available = path.is_file() if source["kind"] == "jsonl" else path.is_dir()
        rows, count = _sample_rows(source_id, path, source["kind"], sample_root) if available else ([], None)
        fields = list(rows[0]) if rows else []
        try:
            mapping = infer_mapping(fields)
        except ValueError:
            mapping = FieldMapping(text_fields=("text",))
            rows = []
        catalog.append({
            "source_id": source_id,
            "path": str(path),
            "adapter": "jsonl" if source["kind"] == "jsonl" else "huggingface_local",
            "available": available,
            "record_count": count,
            "ready": bool(rows),
            "mapping": {
                "text_fields": list(mapping.text_fields), "text_separator": mapping.text_separator,
                "title_field": mapping.title_field, "url_field": mapping.url_field,
                "local_id_field": mapping.local_id_field,
                "metadata_fields": list(mapping.metadata_fields), "record_adapter": mapping.record_adapter,
            },
        })
    return catalog
