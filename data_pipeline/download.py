"""Download a Hugging Face dataset to disk or export selected columns as text."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and save it locally."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset id, for example llamafactory/alpaca_zh.",
    )
    parser.add_argument(
        "--config", default=None, help="Optional dataset configuration name."
    )
    parser.add_argument(
        "--split", default=None, help="Optional split passed to load_dataset()."
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hub revision for reproducible downloads.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory or text file."
    )
    parser.add_argument(
        "--format",
        choices=("disk", "text"),
        default="disk",
        help="Save with datasets.save_to_disk(), or export selected columns as UTF-8 text.",
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=None,
        help="Columns to export in text mode. Defaults to the 'text' column when available.",
    )
    parser.add_argument(
        "--column-separator", default=" ", help="Separator used to join text columns."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing text output.",
    )
    return parser.parse_args()


def _iter_splits(dataset: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(dataset, Mapping):
        for split_name in sorted(dataset):
            yield split_name, dataset[split_name]
    else:
        yield "selected", dataset


def _normalize_field(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def export_text(
    dataset: Any,
    output: Path,
    columns: list[str] | None,
    separator: str,
    overwrite: bool,
) -> int:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --overwrite to replace it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    selected_columns = columns

    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for split_name, split in _iter_splits(dataset):
            if selected_columns is None:
                if "text" not in split.column_names:
                    raise ValueError(
                        f"Split {split_name!r} has no 'text' column. "
                        f"Choose columns with --text-columns. Available: {split.column_names}"
                    )
                selected_columns = ["text"]

            missing = [
                column
                for column in selected_columns
                if column not in split.column_names
            ]
            if missing:
                raise ValueError(
                    f"Split {split_name!r} is missing columns {missing}. Available: {split.column_names}"
                )

            for row in split:
                parts = [_normalize_field(row[column]) for column in selected_columns]
                text = separator.join(part for part in parts if part)
                if text:
                    stream.write(text + "\n")
                    row_count += 1

    return row_count


def main() -> None:
    args = parse_args()

    # Keep --help lightweight and avoid initializing pyarrow before it is needed.
    from datasets import load_dataset

    load_kwargs = {
        "path": args.dataset,
        "name": args.config,
        "split": args.split,
        "revision": args.revision,
    }
    dataset = load_dataset(
        **{key: value for key, value in load_kwargs.items() if value is not None}
    )

    if args.format == "disk":
        if args.output.exists():
            raise FileExistsError(
                f"Output directory already exists: {args.output}. "
                "Choose a new directory instead of mixing dataset versions."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(args.output))
        print(f"Dataset saved to {args.output.resolve()}")
        return

    rows = export_text(
        dataset, args.output, args.text_columns, args.column_separator, args.overwrite
    )
    metadata = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "revision": args.revision,
        "text_columns": args.text_columns,
        "rows": rows,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Exported {rows:,} records to {args.output.resolve()}")


if __name__ == "__main__":
    main()
