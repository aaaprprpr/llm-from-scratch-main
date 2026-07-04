"""Stream text cleaning, train/validation splitting, and optional BPE chunk creation."""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO


CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class PreprocessStats:
    input_files: int = 0
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
                "Pass --overwrite to replace generated chunks."
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
        if self.stream is None or (self.current_bytes > 0 and self.current_bytes + record_bytes > self.target_bytes):
            self._open_next()
        assert self.stream is not None
        self.stream.write(record)
        self.current_bytes += record_bytes

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean UTF-8 text files and reproducibly split them into train/validation text."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/raw/*.txt"],
        help="Input files or glob patterns. Patterns are expanded recursively when they contain **.",
    )
    parser.add_argument("--train-output", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--val-output", type=Path, default=Path("data/val.txt"))
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument(
        "--require-chinese",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only records containing at least one CJK Unified Ideograph.",
    )
    parser.add_argument(
        "--fix-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ftfy when installed. Missing ftfy falls back to identity cleaning.",
    )
    parser.add_argument(
        "--traditional-mode",
        choices=("keep", "drop", "convert"),
        default="keep",
        help="Keep traditional text, drop lines changed by OpenCC, or convert them to simplified Chinese.",
    )
    parser.add_argument(
        "--bpe-chunk-dir",
        type=Path,
        default=None,
        help="Optionally copy accepted training records into fixed-size BPE training chunks.",
    )
    parser.add_argument("--bpe-chunk-size-mb", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_inputs(patterns: Iterable[str]) -> list[Path]:
    resolved: dict[str, Path] = {}
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        for path in matches:
            if path.is_file():
                resolved[str(path.resolve())] = path
    return [resolved[key] for key in sorted(resolved)]


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
            f"--traditional-mode={mode} requires an OpenCC Python package. "
            "Install it or use --traditional-mode keep."
        ) from exc
    converter = OpenCC("t2s")
    return converter.convert


def prepare_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
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
    text = fix_text(raw_line).strip()
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
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if args.min_chars < 1:
        raise ValueError("--min-chars must be positive.")
    if args.bpe_chunk_size_mb < 1:
        raise ValueError("--bpe-chunk-size-mb must be positive.")

    input_paths = resolve_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No input text files matched: {args.input}")

    output_resolved = {args.train_output.resolve(), args.val_output.resolve()}
    input_resolved = {path.resolve() for path in input_paths}
    overlap = output_resolved & input_resolved
    if overlap:
        raise ValueError(f"Input and output paths must be different: {sorted(map(str, overlap))}")

    prepare_outputs((args.train_output, args.val_output), args.overwrite)
    fix_text = make_text_fixer(args.fix_text)
    convert_traditional = make_traditional_converter(args.traditional_mode)
    rng = random.Random(args.seed)
    stats = PreprocessStats(input_files=len(input_paths))
    chunk_writer = None
    if args.bpe_chunk_dir is not None:
        chunk_writer = TextChunkWriter(
            args.bpe_chunk_dir,
            args.bpe_chunk_size_mb * 1024 * 1024,
            args.overwrite,
        )

    try:
        with args.train_output.open("w", encoding="utf-8", newline="\n") as train_stream, args.val_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as val_stream:
            for input_path in input_paths:
                print(f"Processing {input_path}")
                with input_path.open("r", encoding="utf-8") as source:
                    for raw_line in source:
                        stats.total_lines += 1
                        text = clean_line(
                            raw_line,
                            stats,
                            args.min_chars,
                            args.require_chinese,
                            fix_text,
                            args.traditional_mode,
                            convert_traditional,
                        )
                        if text is None:
                            continue

                        stats.accepted += 1
                        if rng.random() < args.train_ratio:
                            train_stream.write(text + "\n")
                            stats.train += 1
                            if chunk_writer is not None:
                                chunk_writer.write(text)
                        else:
                            val_stream.write(text + "\n")
                            stats.validation += 1
    finally:
        if chunk_writer is not None:
            chunk_writer.close()

    metadata = {
        "inputs": [str(path.resolve()) for path in input_paths],
        "train_output": str(args.train_output.resolve()),
        "val_output": str(args.val_output.resolve()),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "min_chars": args.min_chars,
        "require_chinese": args.require_chinese,
        "fix_text": args.fix_text,
        "traditional_mode": args.traditional_mode,
        "stats": asdict(stats),
    }
    metadata_path = args.train_output.parent / "preprocess.meta.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

