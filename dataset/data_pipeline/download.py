"""Fetch the sources selected in configs/data_pipeline.json."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "dataset" / "data_pipeline" / "data" / ".cache" / "huggingface"
DATASET_CACHE = CACHE_ROOT / "datasets"
HUB_CACHE = CACHE_ROOT / "hub"
MANIFEST = PROJECT_ROOT / "dataset" / "data_pipeline" / "data" / "downloads" / "download_manifest.json"
SAMPLE_DIR = PROJECT_ROOT / "dataset" / "data_pipeline" / "dataset_samples"

# Hugging Face reads these at import time. Keep temporary files in this project.
for name, path in {
    "HF_HUB_CACHE": HUB_CACHE,
    "HUGGINGFACE_HUB_CACHE": HUB_CACHE,
    "HF_DATASETS_CACHE": DATASET_CACHE,
    "HF_DATASETS_DOWNLOADED_DATASETS_PATH": DATASET_CACHE / "downloads",
    "HF_DATASETS_EXTRACTED_DATASETS_PATH": DATASET_CACHE / "downloads" / "extracted",
    "HF_XET_CACHE": CACHE_ROOT / "xet",
    "HF_ASSETS_CACHE": CACHE_ROOT / "assets",
    "HF_MODULES_CACHE": CACHE_ROOT / "modules",
}.items():
    os.environ[name] = str(path)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or Path.cwd()).resolve() != SCRIPT_DIR]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_pipeline.config import load_config, project_path, selected_sources
from dataset.data_pipeline.jsonl import read_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:1000] + ("…" if len(value) > 1000 else "")
    if isinstance(value, Mapping):
        return {str(key): _sample_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sample_value(item) for item in value[:20]]
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)


def _write_sample(source_id: str, rows: list[dict]) -> Path:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_DIR / f"{source_id}.sample.json"
    path.write_text(json.dumps({"source": source_id, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _download_jsonl(source: dict[str, Any], path: Path) -> str:
    if path.is_file():
        status = "already_downloaded"
    else:
        if not source.get("repo") or not source.get("filename"):
            raise FileNotFoundError(f"Local JSONL is missing: {path}")
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(hf_hub_download(
            repo_id=source["repo"], repo_type="dataset", filename=source["filename"],
            revision=source.get("revision"), local_dir=str(path.parent), cache_dir=str(HUB_CACHE),
        ))
        if downloaded.resolve() != path.resolve():
            raise RuntimeError(f"Downloaded {downloaded}, expected {path}")
        status = "downloaded"
    actual_sha = _sha256(path)
    expected_sha = source.get("sha256")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected_sha}, got {actual_sha}")
    with path.open("rb") as stream:
        rows = []
        for number, line in enumerate(stream, start=1):
            rows.append({"text": _sample_value(read_text(line, path, number))})
            if len(rows) == 2:
                break
    _write_sample(source["source_id"], rows)
    return status


def _guard_dataset(source: dict[str, Any]) -> None:
    repo = source["repo"]
    expected = {
        "wikimedia/wikipedia": ("20231101.zh", None),
        "opencsg/Fineweb-Edu-Chinese-V2.1": (None, "4_5/*.parquet"),
        "HuggingFaceFW/finewiki": ("zh", "data/zhwiki/*.parquet"),
    }
    if repo not in expected:
        return
    config_name, pattern = expected[repo]
    if source.get("config") != config_name or source["split"] != "train":
        raise ValueError(f"Unsafe subset for {repo}: check config and split")
    if pattern and source.get("data_files") != {"train": pattern}:
        raise ValueError(f"Unsafe file selection for {repo}: expected {pattern}")


def _download_dataset(source: dict[str, Any], path: Path) -> str:
    from datasets import DownloadConfig, load_dataset, load_from_disk

    _guard_dataset(source)
    if path.exists():
        try:
            dataset = load_from_disk(str(path))
        except Exception as exc:
            raise RuntimeError(f"Existing dataset cannot be loaded: {path}") from exc
        status = "already_downloaded"
    else:
        kwargs = {"path": source["repo"], "split": source["split"],
                  "name": source.get("config"), "data_files": source.get("data_files"),
                  "revision": source.get("revision")}
        dataset = load_dataset(
            **{key: value for key, value in kwargs.items() if value is not None},
            cache_dir=str(DATASET_CACHE),
            download_config=DownloadConfig(cache_dir=str(DATASET_CACHE / "downloads")),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(path))
        status = "downloaded"
    sample_dataset = dataset[sorted(dataset)[0]] if isinstance(dataset, Mapping) else dataset
    rows = [_sample_value(sample_dataset[i]) for i in range(min(2, len(sample_dataset)))]
    _write_sample(source["source_id"], rows)
    return status


def download_one(source: dict[str, Any]) -> dict[str, str]:
    path = project_path(source["path"])
    if source["kind"] == "jsonl":
        status = _download_jsonl(source, path)
    else:
        status = _download_dataset(source, path)
    print(f"{source['source_id']}: {status} -> {path}", flush=True)
    return {"source": source["source_id"], "status": status, "path": str(path)}


def main() -> None:
    config = load_config()
    sources = selected_sources(config, "download")
    records = []
    try:
        for source in sources:
            records.append(download_one(source))
    except Exception as exc:
        records.append({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": records,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(record.get("status") == "downloaded" for record in records) and config.require("download", "cleanup_cache") and CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
        print(f"Removed temporary download cache: {CACHE_ROOT}")
    if not sources:
        print("No download sources selected")


if __name__ == "__main__":
    main()
