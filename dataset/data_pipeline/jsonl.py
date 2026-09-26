"""Read validated {"text": str} JSONL without copying a large corpus."""

from __future__ import annotations

import hashlib
import json
from array import array
from pathlib import Path

from tqdm import tqdm


def read_text(line: bytes, path: Path, line_number: int) -> str:
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON at {path}:{line_number}") from exc
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        raise ValueError(f"Expected a string 'text' field at {path}:{line_number}")
    if record["text"] == "":
        raise ValueError(f"Empty 'text' at {path}:{line_number}; input was not modified")
    return record["text"]


class JsonlTextDataset:
    """Index line offsets; retain text only while encoding each requested batch."""

    def __init__(self, path: Path, expected_sha256: str | None = None):
        self.path = Path(path)
        self.offsets = array("Q")
        digest = hashlib.sha256()
        offset = 0
        with self.path.open("rb") as stream, tqdm(
            total=self.path.stat().st_size, unit="B", unit_scale=True,
            desc="Checking text JSONL",
        ) as progress:
            for line_number, line in enumerate(stream, start=1):
                read_text(line, self.path, line_number)
                self.offsets.append(offset)
                digest.update(line)
                offset += len(line)
                progress.update(len(line))
        self._fingerprint = digest.hexdigest()
        if expected_sha256 and self._fingerprint != expected_sha256:
            raise ValueError(
                f"SHA256 mismatch for {self.path}: expected {expected_sha256}, "
                f"got {self._fingerprint}; no bins were written"
            )
        if len(self) < 2:
            raise ValueError("At least two JSONL records are required for train/val splitting")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, key: slice) -> dict[str, list[str]]:
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("Only contiguous record slices are supported")
        texts = []
        if start < stop:
            with self.path.open("rb") as stream:
                stream.seek(self.offsets[start])
                for index in range(start, stop):
                    texts.append(read_text(stream.readline(), self.path, index + 1))
        return {"text": texts}
