"""Normalize raw datasets into complete pretraining text records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import Config

CONFIG_PATH = PROJECT_ROOT / "configs" / "data_pipeline.json"
DEFAULT_OUTPUT = "data_pipeline/data/preprocessed"
LINK_ONLY_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
CONTROL_CHARACTER_TRANSLATION = {
    codepoint: None
    for codepoint in (*range(0x20), *range(0x7F, 0xA0))
    if codepoint not in {0x09, 0x0A}
}
REPETITION_SAMPLE_SIZE = 4096
_ftfy_fix_text = None
_opencc_to_simplified = None
KEEP_REASON = 0
FILTER_REASONS = {
    1: "empty_or_invalid",
    2: "link_only",
    3: "no_letters",
    4: "corrupted",
    5: "repetitive",
}


@dataclass
class PreprocessStats:
    dataset_sources: int = 0
    dataset_rows: int = 0
    duplicates: int = 0
    accepted: int = 0


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    text = text.translate(CONTROL_CHARACTER_TRANSLATION)
    return text.strip()


def first_text(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def join_parts(*parts: str) -> str | None:
    texts = [
        text
        for part in parts
        if isinstance(part, str) and (text := part.strip())
    ]
    if not texts:
        return None
    return "\n\n".join(texts)


def clean_record(
    text: str | None,
    max_repetition_ratio: float,
    fix_text: bool,
    convert_to_simplified: bool,
) -> tuple[str | None, int]:
    if not text:
        return None, 1

    if fix_text:
        global _ftfy_fix_text
        if _ftfy_fix_text is None:
            import ftfy

            _ftfy_fix_text = ftfy.fix_text
        text = _ftfy_fix_text(text)
    text = normalize_text(text)
    if not text:
        return None, 1
    if convert_to_simplified:
        global _opencc_to_simplified
        if _opencc_to_simplified is None:
            try:
                from opencc import OpenCC
            except ImportError as exc:
                raise RuntimeError(
                    "preprocess.convert_to_simplified=true requires "
                    "opencc-python-reimplemented"
                ) from exc
            _opencc_to_simplified = OpenCC("tw2sp")
        text = _opencc_to_simplified.convert(text)
    if LINK_ONLY_PATTERN.fullmatch(text):
        return None, 2

    compact_length = 0
    replacement_count = 0
    has_letter = False
    repetition_sample = []
    for character in text:
        if character.isspace():
            continue
        compact_length += 1
        replacement_count += character == "\ufffd"
        has_letter = has_letter or character.isalpha()
        if len(repetition_sample) < REPETITION_SAMPLE_SIZE and character.isalnum():
            repetition_sample.append(character.casefold())

    if compact_length == 0:
        return None, 1
    if not has_letter:
        return None, 3
    if replacement_count / compact_length > 0.02:
        return None, 4

    if len(repetition_sample) >= 20:
        most_common_count = Counter(repetition_sample).most_common(1)[0][1]
        if most_common_count / len(repetition_sample) > max_repetition_ratio:
            return None, 5
        sample_text = "".join(repetition_sample)
        max_pattern_length = min(16, len(sample_text) // 4)
        for pattern_length in range(2, max_pattern_length + 1):
            if sample_text[pattern_length:] == sample_text[:-pattern_length]:
                return None, 5

    return text, KEEP_REASON


def adapt_plain_text(row: Mapping[str, Any]) -> str | None:
    return join_parts(
        first_text(row, "title"),
        first_text(row, "text", "content"),
    )


def adapt_classification_text(row: Mapping[str, Any]) -> str | None:
    return join_parts(first_text(row, "text", "content"))


def adapt_instruction_input_output(row: Mapping[str, Any]) -> str | None:
    instruction = first_text(row, "instruction", "prompt", "question")
    input_text = first_text(row, "input", "context")
    output = first_text(row, "output", "answer", "response")
    return join_parts(instruction, input_text, output)


def adapt_question_answer(row: Mapping[str, Any]) -> str | None:
    question = first_text(row, "question", "query", "prompt")
    answer = first_text(row, "answer", "output", "response")
    return join_parts(question, answer)


def adapt_question_answer_optional_think(
    row: Mapping[str, Any],
) -> str | None:
    question = first_text(row, "question", "query", "prompt")
    think = first_text(row, "think", "reasoning")
    answer = first_text(row, "answer", "output", "response")
    return join_parts(question, think, answer)


def adapt_conversation(row: Mapping[str, Any]) -> str | None:
    conversation = row.get("conversations")
    if conversation is None:
        conversation = row.get("conversation")

    if isinstance(conversation, str):
        return join_parts(conversation)
    if not isinstance(conversation, list | tuple):
        return None

    contents = []
    for turn in conversation:
        if isinstance(turn, Mapping):
            content = first_text(turn, "content", "value", "text")
        elif isinstance(turn, str):
            content = turn.strip()
        else:
            content = ""
        if content:
            contents.append(content)
    return join_parts(*contents)


def adapt_tieba_thread(row: Mapping[str, Any]) -> str | None:
    title = first_text(row, "标题", "title")
    author_content = first_text(row, "楼主内容", "content", "text")
    raw_replies = row.get("回复列表")
    if raw_replies is None:
        raw_replies = row.get("replies")

    replies = []
    if isinstance(raw_replies, list | tuple):
        for reply in raw_replies:
            if isinstance(reply, Mapping):
                text = first_text(reply, "content", "value", "text")
            else:
                text = reply.strip() if isinstance(reply, str) else ""
            if text:
                replies.append(text)
    elif isinstance(raw_replies, str):
        text = raw_replies.strip()
        if text:
            replies.append(text)

    return join_parts(title, author_content, *replies)


ADAPTERS: dict[str, Callable[[Mapping[str, Any]], str | None]] = {
    "plain_text": adapt_plain_text,
    "classification_text": adapt_classification_text,
    "instruction_input_output": adapt_instruction_input_output,
    "question_answer": adapt_question_answer,
    "question_answer_optional_think": adapt_question_answer_optional_think,
    "sharegpt_conversations": adapt_conversation,
    "openai_role_content_conversation": adapt_conversation,
    "tieba_thread": adapt_tieba_thread,
}

ADAPTER_COLUMNS = {
    "plain_text": {"title", "text", "content"},
    "classification_text": {"text", "content"},
    "instruction_input_output": {
        "instruction",
        "prompt",
        "question",
        "input",
        "context",
        "output",
        "answer",
        "response",
    },
    "question_answer": {
        "question",
        "query",
        "prompt",
        "answer",
        "output",
        "response",
    },
    "question_answer_optional_think": {
        "question",
        "query",
        "prompt",
        "think",
        "reasoning",
        "answer",
        "output",
        "response",
    },
    "sharegpt_conversations": {"conversations", "conversation"},
    "openai_role_content_conversation": {"conversation", "conversations"},
    "tieba_thread": {
        "标题",
        "title",
        "楼主内容",
        "content",
        "text",
        "回复列表",
        "replies",
    },
}


def select_dataset_sources(
    downloads: list[dict[str, Any]],
    source_settings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    downloads_by_id = {source["source_id"]: source for source in downloads}
    selected = []
    for setting in source_settings:
        if not setting.get("enabled", False):
            continue
        source_id = setting["source_id"]
        if source_id not in downloads_by_id:
            raise KeyError(f"Unknown dataset source_id in preprocess: {source_id}")
        selected.append({**downloads_by_id[source_id], **setting})
    return selected


def iter_splits(dataset: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(dataset, Mapping):
        for split_name in sorted(dataset):
            yield split_name, dataset[split_name]
    else:
        yield "selected", dataset


def clean_batch(
    batch: Mapping[str, list[Any]],
    rank: int | None,
    adapter_name: str,
    max_repetition_ratio: float,
    fix_text: bool,
    convert_to_simplified: bool,
    rejected_dir: str,
) -> dict[str, list[Any]]:
    adapter = ADAPTERS[adapter_name]
    batch_size = len(next(iter(batch.values()), []))
    texts = []
    digests = []
    rejected = []

    for index in range(batch_size):
        row = {name: values[index] for name, values in batch.items()}
        original_text = adapter(row)
        text, reason = clean_record(
            original_text,
            max_repetition_ratio,
            fix_text,
            convert_to_simplified,
        )
        if text is None:
            rejected.append(
                {
                    "reason": FILTER_REASONS[reason],
                    "text": original_text or "",
                }
            )
            continue

        texts.append(text)
        digests.append(
            hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        )

    if rejected:
        rejected_path = Path(rejected_dir) / f"worker-{rank or 0:05d}.jsonl"
        with rejected_path.open("a", encoding="utf-8") as stream:
            for record in rejected:
                stream.write(json.dumps(record, ensure_ascii=False))
                stream.write("\n")

    return {"text": texts, "_digest": digests}


def clean_source(
    source: dict[str, Any],
    missing_policy: str,
    max_repetition_ratio: float,
    fix_text: bool,
    convert_to_simplified: bool,
    workers: int,
    map_batch_size: int,
    stats: PreprocessStats,
    cache_dir: Path,
    rejected_dir: Path,
) -> list[Any]:
    from datasets import Features, Value, load_from_disk

    input_path = PROJECT_ROOT / source["output"]
    if not input_path.exists():
        message = (
            f"Downloaded dataset does not exist for {source['source_id']}: "
            f"{input_path}"
        )
        if missing_policy == "skip":
            print(f"Warning: {message}; skipped.")
            return []
        raise FileNotFoundError(message)

    adapter_name = source.get("adapter", "plain_text")
    if adapter_name not in ADAPTERS:
        raise KeyError(
            f"Unknown adapter {adapter_name!r} for source {source['source_id']}"
        )

    dataset = load_from_disk(str(input_path))
    cleaned_splits = []
    output_features = Features(
        {
            "text": Value("string"),
            "_digest": Value("binary"),
        }
    )
    num_proc = workers if workers > 1 else None

    for split_index, (split_name, split) in enumerate(iter_splits(dataset)):
        print(f"Processing dataset {source['source_id']} split {split_name}")
        stats.dataset_rows += len(split)
        input_columns = [
            column
            for column in split.column_names
            if column in ADAPTER_COLUMNS[adapter_name]
        ]
        if not input_columns:
            raise KeyError(
                f"Dataset {source['source_id']} has none of the columns expected "
                f"by adapter {adapter_name!r}."
            )
        split = split.select_columns(input_columns)

        cache_prefix = f"{source['source_id']}-{split_index:03d}"
        mapped = split.map(
            clean_batch,
            batched=True,
            batch_size=map_batch_size,
            num_proc=num_proc,
            with_rank=True,
            fn_kwargs={
                "adapter_name": adapter_name,
                "max_repetition_ratio": max_repetition_ratio,
                "fix_text": fix_text,
                "convert_to_simplified": convert_to_simplified,
                "rejected_dir": str(rejected_dir),
            },
            remove_columns=split.column_names,
            features=output_features,
            cache_file_name=str(cache_dir / f"{cache_prefix}-map.arrow"),
            desc=f"Cleaning {source['source_id']}:{split_name}",
        )
        if len(mapped):
            cleaned_splits.append(mapped)

    return cleaned_splits


def build_dataset(
    sources: list[dict[str, Any]],
    missing_policy: str,
    deduplicate: bool,
    stats: PreprocessStats,
    cache_dir: Path,
    max_repetition_ratio: float,
    fix_text: bool,
    convert_to_simplified: bool,
    workers: int,
    map_batch_size: int,
    rejected_dir: Path,
):
    from datasets import Dataset, Features, Value, concatenate_datasets

    cleaned_parts = []
    for source in sources:
        cleaned_parts.extend(
            clean_source(
                source,
                missing_policy,
                max_repetition_ratio,
                fix_text,
                convert_to_simplified,
                workers,
                map_batch_size,
                stats,
                cache_dir,
                rejected_dir,
            )
        )
    if not cleaned_parts:
        raise ValueError("No valid pretraining text records remain after cleaning.")

    cleaned_dataset = concatenate_datasets(cleaned_parts)
    if not deduplicate:
        stats.accepted = len(cleaned_dataset)
        return cleaned_dataset.remove_columns("_digest")

    def generate_records():
        seen_hashes: set[bytes] = set()
        duplicate_path = rejected_dir / "duplicates.jsonl"
        with duplicate_path.open("a", encoding="utf-8") as rejected_stream:
            for text_batch in cleaned_dataset.iter(batch_size=10_000):
                for text, digest in zip(
                    text_batch["text"],
                    text_batch["_digest"],
                ):
                    if digest in seen_hashes:
                        stats.duplicates += 1
                        rejected_stream.write(
                            json.dumps(
                                {"reason": "duplicate", "text": text},
                                ensure_ascii=False,
                            )
                        )
                        rejected_stream.write("\n")
                        continue
                    seen_hashes.add(digest)

                    stats.accepted += 1
                    yield {"text": text}

    return Dataset.from_generator(
        generate_records,
        features=Features({"text": Value("string")}),
        cache_dir=str(cache_dir),
    )


def write_report(
    stats: PreprocessStats,
    rejected_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    rejected_files = sorted(rejected_dir.glob("*.jsonl"))
    filtered_by_reason: Counter[str] = Counter()
    for rejected_file in rejected_files:
        with rejected_file.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    filtered_by_reason[json.loads(line)["reason"]] += 1

    filtered_by_reason = {
        reason: filtered_by_reason.get(reason, 0)
        for reason in (*FILTER_REASONS.values(), "duplicate")
    }
    filtered = sum(filtered_by_reason.values())
    if stats.accepted + filtered != stats.dataset_rows:
        raise RuntimeError(
            "Preprocess statistics are inconsistent: "
            f"{stats.accepted} accepted + {filtered} filtered != "
            f"{stats.dataset_rows} input rows."
        )
    summary = {
        "dataset_sources": stats.dataset_sources,
        "input_records": stats.dataset_rows,
        "accepted_records": stats.accepted,
        "filtered_records": filtered,
        "filtered_by_reason": filtered_by_reason,
    }
    with output_path.open("w", encoding="utf-8") as report:
        report.write('{\n  "summary": ')
        report.write(json.dumps(summary, ensure_ascii=False, indent=2))
        report.write(',\n  "records": [')
        first_record = True
        for rejected_file in rejected_files:
            with rejected_file.open(encoding="utf-8") as stream:
                for line in stream:
                    record = line.strip()
                    if not record:
                        continue
                    report.write("\n    " if first_record else ",\n    ")
                    report.write(record)
                    first_record = False
        if not first_record:
            report.write("\n  ")
        report.write("]\n}\n")
    return summary


def main() -> None:
    root_config = Config(CONFIG_PATH)
    config = root_config.require("preprocess")
    downloads = root_config.require("downloads")

    missing_policy = config.get("missing_dataset_policy", "error")
    if missing_policy not in {"error", "skip"}:
        raise ValueError("missing_dataset_policy must be one of: error, skip.")

    sources = select_dataset_sources(
        downloads,
        config.get("dataset_sources", []),
    )
    if not sources:
        raise ValueError("No preprocess dataset sources are enabled.")

    output_path = PROJECT_ROOT / config.get("output", DEFAULT_OUTPUT)
    overwrite = config.get("overwrite", False)
    deduplicate = config.get("deduplicate", True)
    max_repetition_ratio = config.get("max_repetition_ratio", 0.8)
    fix_text = config.get("fix_text", True)
    convert_to_simplified = config.get("convert_to_simplified", False)
    workers = config.get("workers", "auto")
    map_batch_size = config.get("map_batch_size", 1000)

    if not 0.0 < max_repetition_ratio <= 1.0:
        raise ValueError("max_repetition_ratio must be between 0 and 1.")
    if workers == "auto":
        workers = max(1, (os.cpu_count() or 2) - 1)
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer or 'auto'.")
    if not isinstance(map_batch_size, int) or map_batch_size < 1:
        raise ValueError("map_batch_size must be positive.")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Set overwrite=true in configs/data_pipeline.json to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    report_path = output_path.with_name(f"{output_path.name}.report.json")
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    if temporary_report.exists():
        temporary_report.unlink()

    stats = PreprocessStats(dataset_sources=len(sources))
    try:
        with tempfile.TemporaryDirectory(
            prefix=".preprocess-cache-",
            dir=output_path.parent,
        ) as cache_dir:
            cache_path = Path(cache_dir)
            rejected_dir = cache_path / "rejected"
            rejected_dir.mkdir()
            dataset = build_dataset(
                sources,
                missing_policy,
                deduplicate,
                stats,
                cache_path,
                max_repetition_ratio,
                fix_text,
                convert_to_simplified,
                workers,
                map_batch_size,
                rejected_dir,
            )
            dataset.save_to_disk(str(temporary_output))
            report = write_report(stats, rejected_dir, temporary_report)

        if output_path.exists():
            shutil.rmtree(output_path)
        temporary_output.replace(output_path)
        temporary_report.replace(report_path)
    except BaseException:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        if temporary_report.exists():
            temporary_report.unlink()
        raise

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {len(dataset):,} records to {output_path.resolve()}")


if __name__ == "__main__":
    main()
