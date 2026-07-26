"""Stream text cleaning, train/validation splitting, and optional BPE chunk creation."""

from __future__ import annotations

import glob
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import Config

CONFIG_PATH = PROJECT_ROOT / "configs" / "data_pipeline.json"
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class PreprocessStats:
    input_files: int = 0
    raw_text_files: int = 0
    dataset_sources: int = 0
    dataset_rows: int = 0
    total_lines: int = 0
    empty_lines: int = 0
    too_short: int = 0
    no_chinese: int = 0
    traditional_dropped: int = 0
    accepted: int = 0
    train: int = 0
    validation: int = 0


class TextChunkWriter:
    """Write complete UTF-8 records into approximately fixed-size text chunks."""

    def __init__(self, output_dir: Path, target_bytes: int, overwrite: bool):
        self.output_dir = output_dir
        self.target_bytes = target_bytes
        self.index = 0
        self.current_bytes = 0
        self.stream: TextIO | None = None

        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(output_dir.glob("chunk_*.txt"))
        if existing and not overwrite:
            raise FileExistsError(
                f"BPE chunk directory already contains chunk_*.txt: {output_dir}. "
                "Set overwrite=true in configs/data_pipeline.json to replace generated chunks."
            )
        if overwrite:
            for path in existing:
                path.unlink()

    def _open_next(self) -> None:
        if self.stream is not None:
            self.stream.close()
        self.index += 1
        path = self.output_dir / f"chunk_{self.index:05d}.txt"
        self.stream = path.open("w", encoding="utf-8", newline="\n")
        self.current_bytes = 0

    def write(self, text: str) -> None:
        record = text + "\n"
        record_bytes = len(record.encode("utf-8"))
        if self.stream is None or (
            self.current_bytes > 0
            and self.current_bytes + record_bytes > self.target_bytes
        ):
            self._open_next()
        assert self.stream is not None
        self.stream.write(record)
        self.current_bytes += record_bytes

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def resolve_inputs(patterns: Iterable[str]) -> list[Path]:
    resolved: dict[str, Path] = {}
    for pattern in patterns:
        pattern_path = Path(pattern)
        search_pattern = (
            pattern_path if pattern_path.is_absolute() else PROJECT_ROOT / pattern_path
        )
        matches = [Path(path) for path in glob.glob(str(search_pattern), recursive=True)]
        if not matches and search_pattern.is_file():
            matches = [search_pattern]
        for path in matches:
            if path.is_file():
                resolved[str(path.resolve())] = path
    return [resolved[key] for key in sorted(resolved)]


def _iter_splits(dataset: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(dataset, Mapping):
        for split_name in sorted(dataset):
            yield split_name, dataset[split_name]
    else:
        yield "selected", dataset


def _compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, list | tuple | dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def _first_field(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        text = _compact_text(value)
        if text:
            return text
    return ""


ROLE_NAMES = {
    "human": "用户",
    "user": "用户",
    "assistant": "助手",
    "gpt": "助手",
    "bot": "助手",
    "system": "系统",
}


def _format_turn(turn: Any) -> str:
    if isinstance(turn, Mapping):
        role = _compact_text(
            turn.get("role")
            or turn.get("from")
            or turn.get("speaker")
            or turn.get("name")
        )
        content = _first_field(turn, "content", "value", "text")
        if not content:
            content = _compact_text(turn)
        if role:
            return f"{ROLE_NAMES.get(role.lower(), role)}：{content}"
        return content
    return _compact_text(turn)


def _format_conversation(value: Any) -> str:
    if isinstance(value, str):
        return _compact_text(value)
    if isinstance(value, list | tuple):
        return " ".join(part for part in (_format_turn(turn) for turn in value) if part)
    return _compact_text(value)


def adapt_plain_text(row: Mapping[str, Any]) -> str:
    return _first_field(row, "text")


def adapt_classification_text(row: Mapping[str, Any]) -> str:
    return _first_field(row, "text", "content")


def adapt_instruction_input_output(row: Mapping[str, Any]) -> str:
    instruction = _first_field(row, "instruction", "prompt", "question")
    input_text = _first_field(row, "input", "context")
    output = _first_field(row, "output", "answer", "response")
    parts = []
    if instruction:
        parts.append(f"指令：{instruction}")
    if input_text:
        parts.append(f"输入：{input_text}")
    if output:
        parts.append(f"回答：{output}")
    return " ".join(parts)


def adapt_question_answer(row: Mapping[str, Any]) -> str:
    question = _first_field(row, "question", "query", "prompt")
    answer = _first_field(row, "answer", "output", "response")
    parts = []
    if question:
        parts.append(f"问题：{question}")
    if answer:
        parts.append(f"回答：{answer}")
    return " ".join(parts)


def adapt_question_answer_optional_think(row: Mapping[str, Any]) -> str:
    question = _first_field(row, "question", "query", "prompt")
    think = _first_field(row, "think", "reasoning")
    answer = _first_field(row, "answer", "output", "response")
    parts = []
    if question:
        parts.append(f"问题：{question}")
    if think:
        parts.append(f"思考：{think}")
    if answer:
        parts.append(f"回答：{answer}")
    return " ".join(parts)


def adapt_sharegpt_conversations(row: Mapping[str, Any]) -> str:
    return _format_conversation(row.get("conversations") or row.get("conversation"))


def adapt_openai_role_content_conversation(row: Mapping[str, Any]) -> str:
    return _format_conversation(row.get("conversation") or row.get("conversations"))


def adapt_tieba_thread(row: Mapping[str, Any]) -> str:
    title = _first_field(row, "标题", "title")
    author_content = _first_field(row, "楼主内容", "content", "text")
    replies = row.get("回复列表") or row.get("replies") or []
    parts = []
    if title:
        parts.append(f"标题：{title}")
    if author_content:
        parts.append(f"楼主：{author_content}")
    if isinstance(replies, list | tuple):
        for reply in replies:
            reply_text = _compact_text(reply)
            if reply_text:
                parts.append(f"回复：{reply_text}")
    else:
        reply_text = _compact_text(replies)
        if reply_text:
            parts.append(f"回复：{reply_text}")
    return " ".join(parts)


ADAPTERS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "plain_text": adapt_plain_text,
    "classification_text": adapt_classification_text,
    "instruction_input_output": adapt_instruction_input_output,
    "question_answer": adapt_question_answer,
    "question_answer_optional_think": adapt_question_answer_optional_think,
    "sharegpt_conversations": adapt_sharegpt_conversations,
    "openai_role_content_conversation": adapt_openai_role_content_conversation,
    "tieba_thread": adapt_tieba_thread,
}


def select_dataset_sources(
    downloads: list[dict[str, Any]],
    source_settings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    downloads_by_id = {source["source_id"]: source for source in downloads}
    selected: list[dict[str, Any]] = []
    for setting in source_settings:
        if not setting.get("enabled", False):
            continue
        source_id = setting["source_id"]
        if source_id not in downloads_by_id:
            raise KeyError(f"Unknown dataset source_id in preprocess: {source_id}")
        selected.append({**downloads_by_id[source_id], **setting})
    return selected


def iter_dataset_texts(
    source: dict[str, Any],
    missing_policy: str,
) -> Iterable[str]:
    from datasets import load_from_disk

    output = PROJECT_ROOT / source["output"]
    if not output.exists():
        message = f"Downloaded dataset does not exist for {source['source_id']}: {output}"
        if missing_policy == "skip":
            print(f"Warning: {message}; skipped.")
            return
        raise FileNotFoundError(message)

    adapter_name = source.get("adapter") or "plain_text"
    adapter = ADAPTERS.get(adapter_name)
    if adapter is None:
        raise KeyError(f"Unknown adapter {adapter_name!r} for source {source['source_id']}")

    dataset = load_from_disk(str(output))
    for split_name, split in _iter_splits(dataset):
        print(f"Processing dataset {source['source_id']} split {split_name}")
        for row in split:
            yield adapter(row)


def make_text_fixer(enabled: bool) -> Callable[[str], str]:
    if not enabled:
        return lambda text: text
    try:
        import ftfy
    except ImportError:
        print("Warning: ftfy is not installed; continuing without Unicode repair.")
        return lambda text: text
    return ftfy.fix_text


def make_traditional_converter(mode: str) -> Callable[[str], str]:
    if mode == "keep":
        return lambda text: text
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise RuntimeError(
            f"traditional_mode={mode} requires an OpenCC Python package. "
            "Install it or set traditional_mode=keep in configs/data_pipeline.json."
        ) from exc
    converter = OpenCC("t2s")
    return converter.convert


def prepare_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Set overwrite=true in configs/data_pipeline.json to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)


def clean_line(
    raw_line: str,
    stats: PreprocessStats,
    min_chars: int,
    require_chinese: bool,
    fix_text: Callable[[str], str],
    traditional_mode: str,
    convert_traditional: Callable[[str], str],
) -> str | None:
    text = re.sub(r"\s+", " ", fix_text(raw_line)).strip()
    if not text:
        stats.empty_lines += 1
        return None
    if len(text) < min_chars:
        stats.too_short += 1
        return None
    if require_chinese and CHINESE_PATTERN.search(text) is None:
        stats.no_chinese += 1
        return None

    converted = convert_traditional(text)
    if traditional_mode == "drop" and converted != text:
        stats.traditional_dropped += 1
        return None
    if traditional_mode == "convert":
        text = converted
    return text


def main() -> None:
    root_config = Config(CONFIG_PATH)
    config = root_config.require("preprocess")
    downloads = root_config.require("downloads")
    train_output = PROJECT_ROOT / config["train_output"]
    val_output = PROJECT_ROOT / config["val_output"]
    bpe_chunk_dir = (
        PROJECT_ROOT / config["bpe_chunk_dir"]
        if config["bpe_chunk_dir"] is not None
        else None
    )

    if not 0.0 < config["train_ratio"] < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if config["min_chars"] < 1:
        raise ValueError("min_chars must be positive.")
    if config["bpe_chunk_size_mb"] < 1:
        raise ValueError("bpe_chunk_size_mb must be positive.")
    if config["traditional_mode"] not in {"keep", "convert", "drop"}:
        raise ValueError("traditional_mode must be one of: keep, convert, drop.")

    if config.get("missing_dataset_policy", "error") not in {"error", "skip"}:
        raise ValueError("missing_dataset_policy must be one of: error, skip.")

    raw_text_inputs = config.get("raw_text_inputs", config.get("input", []))
    input_paths = resolve_inputs(raw_text_inputs)
    dataset_sources = select_dataset_sources(
        downloads,
        config.get("dataset_sources", []),
    )
    if not input_paths and not dataset_sources:
        raise FileNotFoundError(
            "No preprocess inputs enabled. Enable preprocess.dataset_sources or set raw_text_inputs."
        )

    output_resolved = {train_output.resolve(), val_output.resolve()}
    input_resolved = {path.resolve() for path in input_paths}
    overlap = output_resolved & input_resolved
    if overlap:
        raise ValueError(
            f"Input and output paths must be different: {sorted(map(str, overlap))}"
        )

    prepare_outputs((train_output, val_output), config["overwrite"])
    fix_text = make_text_fixer(config["fix_text"])
    convert_traditional = make_traditional_converter(config["traditional_mode"])
    rng = random.Random(config["seed"])
    stats = PreprocessStats(
        input_files=len(input_paths) + len(dataset_sources),
        raw_text_files=len(input_paths),
        dataset_sources=len(dataset_sources),
    )
    chunk_writer = None
    if bpe_chunk_dir is not None:
        chunk_writer = TextChunkWriter(
            bpe_chunk_dir,
            config["bpe_chunk_size_mb"] * 1024 * 1024,
            config["overwrite"],
        )

    try:
        with train_output.open("w", encoding="utf-8", newline="\n") as train_stream, val_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as val_stream:
            def consume_text(raw_text: str) -> None:
                stats.total_lines += 1
                text = clean_line(
                    raw_text,
                    stats,
                    config["min_chars"],
                    config["require_chinese"],
                    fix_text,
                    config["traditional_mode"],
                    convert_traditional,
                )
                if text is None:
                    return

                stats.accepted += 1
                if rng.random() < config["train_ratio"]:
                    train_stream.write(text + "\n")
                    stats.train += 1
                    if chunk_writer is not None:
                        chunk_writer.write(text)
                else:
                    val_stream.write(text + "\n")
                    stats.validation += 1

            for dataset_source in dataset_sources:
                for raw_text in iter_dataset_texts(
                    dataset_source,
                    config.get("missing_dataset_policy", "error"),
                ):
                    stats.dataset_rows += 1
                    consume_text(raw_text)

            for input_path in input_paths:
                print(f"Processing {input_path}")
                with input_path.open("r", encoding="utf-8") as source:
                    for raw_line in source:
                        consume_text(raw_line)
    finally:
        if chunk_writer is not None:
            chunk_writer.close()

    metadata = {
        "raw_text_inputs": [str(path.resolve()) for path in input_paths],
        "dataset_sources": [
            {
                "source_id": source["source_id"],
                "output": str((PROJECT_ROOT / source["output"]).resolve()),
                "adapter": source.get("adapter"),
            }
            for source in dataset_sources
        ],
        "train_output": str(train_output.resolve()),
        "val_output": str(val_output.resolve()),
        "train_ratio": config["train_ratio"],
        "seed": config["seed"],
        "min_chars": config["min_chars"],
        "require_chinese": config["require_chinese"],
        "fix_text": config["fix_text"],
        "traditional_mode": config["traditional_mode"],
        "stats": asdict(stats),
    }
    metadata_path = train_output.parent / "preprocess.meta.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
