"""Download configured datasets and write small structure samples."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_CACHE_ROOT = PROJECT_ROOT / "data_pipeline" / "data" / ".cache" / "huggingface"
DATASET_CACHE_DIR = DOWNLOAD_CACHE_ROOT / "datasets"
RAW_DOWNLOAD_CACHE_DIR = DATASET_CACHE_DIR / "downloads"

# Set cache paths before importing datasets / huggingface_hub: both libraries
# read these environment variables at import time. Override inherited cache
# locations for this downloader process, while keeping HF_HOME/token settings.
for _cache_variable, _cache_path in {
    "HF_HUB_CACHE": DOWNLOAD_CACHE_ROOT / "hub",
    "HUGGINGFACE_HUB_CACHE": DOWNLOAD_CACHE_ROOT / "hub",
    "HF_DATASETS_CACHE": DATASET_CACHE_DIR,
    "HF_DATASETS_DOWNLOADED_DATASETS_PATH": RAW_DOWNLOAD_CACHE_DIR,
    "HF_DATASETS_EXTRACTED_DATASETS_PATH": RAW_DOWNLOAD_CACHE_DIR / "extracted",
    "HF_XET_CACHE": DOWNLOAD_CACHE_ROOT / "xet",
    "HF_ASSETS_CACHE": DOWNLOAD_CACHE_ROOT / "assets",
    "HF_MODULES_CACHE": DOWNLOAD_CACHE_ROOT / "modules",
}.items():
    os.environ[_cache_variable] = str(_cache_path)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import Config

CONFIG_PATH = PROJECT_ROOT / "configs" / "data_pipeline.json"
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data_pipeline"
    / "data"
    / "downloads"
    / "download_manifest.json"
)
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data_pipeline" / "dataset_samples"


def _project_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _iter_splits(dataset: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(dataset, Mapping):
        for split_name in sorted(dataset):
            yield split_name, dataset[split_name]
    else:
        yield "selected", dataset


def _safe_json_value(value: Any, max_chars: int) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...<truncated>"
    if isinstance(value, Mapping):
        return {
            str(key): _safe_json_value(item, max_chars)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_safe_json_value(item, max_chars) for item in value[:20]]
    return _safe_json_value(str(value), max_chars)


def dataset_summary(dataset: Any) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split_name, split in _iter_splits(dataset):
        row_count = getattr(split, "num_rows", None)
        splits[split_name] = {
            "num_rows": int(row_count) if row_count is not None else None,
            "columns": list(getattr(split, "column_names", []) or []),
            "features": str(getattr(split, "features", "")),
        }
    return {"splits": splits}


def dataset_sample(dataset: Any, row_limit: int, max_chars: int) -> dict[str, Any]:
    samples: dict[str, list[Any]] = {}
    for split_name, split in _iter_splits(dataset):
        rows: list[Any] = []
        for index, row in enumerate(split):
            if index >= row_limit:
                break
            rows.append(_safe_json_value(row, max_chars))
        samples[split_name] = rows
    return samples


def write_dataset_sample(
    source: dict[str, Any],
    dataset: Any,
    sample_dir: Path,
    row_limit: int,
    max_chars: int,
) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{source['source_id']}.sample.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_id": source.get("source_id"),
        "dataset": source.get("dataset"),
        "config": source.get("config"),
        "split": source.get("split"),
        "data_files": source.get("data_files"),
        "adapter": source.get("adapter"),
        "expected_columns": source.get("expected_columns"),
        "summary": dataset_summary(dataset),
        "sample_rows": dataset_sample(dataset, row_limit, max_chars),
    }
    sample_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sample_path


def _source_record(
    source: dict[str, Any],
    status: str,
    output: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configured_output = _project_path(source.get("output"))
    record = {
        "source_id": source.get("source_id"),
        "enabled": source.get("enabled", False),
        "source_type": source.get("source_type", "huggingface"),
        "purpose": source.get("purpose"),
        "description": source.get("description"),
        "source_url": source.get("source_url"),
        "dataset": source.get("dataset"),
        "config": source.get("config"),
        "split": source.get("split"),
        "data_files": source.get("data_files"),
        "revision": source.get("revision"),
        "license": source.get("license"),
        "format": source.get("format"),
        "expected_columns": source.get("expected_columns"),
        "adapter": source.get("adapter"),
        "size_note": source.get("size_note"),
        "safety_note": source.get("safety_note"),
        "source_path": source.get("source_path"),
        "output": (
            str(output.resolve())
            if output is not None
            else str(configured_output.resolve())
            if configured_output is not None
            else None
        ),
        "status": status,
        "notes": source.get("notes"),
    }
    if extra:
        record.update(extra)
    return record


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(CONFIG_PATH.resolve()),
        "sources": records,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Download manifest written to {path.resolve()}")


def load_local_dataset(output: Path) -> Any:
    from datasets import load_from_disk

    return load_from_disk(str(output))


def download_cache_size() -> int:
    """当前下载缓存占用的字节数。

    hub 缓存里 snapshots/ 是指向 blobs/ 的软链接，跟着软链接统计会把同一份
    文件数两遍，所以软链接一律跳过。
    """
    if not DOWNLOAD_CACHE_ROOT.exists():
        return 0
    total = 0
    for path in DOWNLOAD_CACHE_ROOT.rglob("*"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            total += path.stat().st_size
        except OSError:
            continue
    return total


def cleanup_download_cache() -> None:
    """删除下载缓存。

    load_dataset() 必须先把远端文件落到一个中转目录，save_to_disk() 才能写出
    可直接使用的副本。那个中转目录就是 DOWNLOAD_CACHE_ROOT；所有启用源都
    save_to_disk 成功之后，它不再被任何东西读取，唯一的用途就是占地方。
    """
    if not DOWNLOAD_CACHE_ROOT.exists():
        print(f"Download cache already empty: {DOWNLOAD_CACHE_ROOT}")
        return
    size = download_cache_size()
    shutil.rmtree(DOWNLOAD_CACHE_ROOT)
    print(
        f"Download cache removed: {DOWNLOAD_CACHE_ROOT} "
        f"(freed {size / 2**30:.2f} GiB)"
    )


def check_manual_source(source: dict[str, Any]) -> dict[str, Any]:
    source_path = _project_path(source.get("source_path"))
    output = _project_path(source.get("output"))
    source_exists = source_path.exists() if source_path is not None else False
    output_exists = output.exists() if output is not None else False
    status = "manual_available" if source_exists or output_exists else "manual_missing"
    return _source_record(
        source,
        status,
        output,
        {
            "source_path_resolved": str(source_path.resolve()) if source_path else None,
            "source_exists": source_exists,
            "output_exists": output_exists,
        },
    )


def guard_dangerous_configs(source: dict[str, Any]) -> None:
    if source.get("dataset") == "wikimedia/wikipedia":
        if source.get("config") != "20231101.zh":
            raise ValueError(
                "wikimedia/wikipedia must use config='20231101.zh'. "
                "Do not download the full wikipedia dataset by accident."
            )
        if source.get("split") != "train":
            raise ValueError("wikimedia/wikipedia must use split='train'.")

    restricted_subsets = {
        "opencsg/Fineweb-Edu-Chinese-V2.1": (None, "4_5/*.parquet"),
        "HuggingFaceFW/finewiki": ("zh", "data/zhwiki/*.parquet"),
    }
    if source.get("dataset") in restricted_subsets:
        config_name, pattern = restricted_subsets[source["dataset"]]
        if source.get("config") != config_name:
            raise ValueError(
                f"{source['dataset']} must use config={config_name!r}."
            )
        if source.get("split") != "train":
            raise ValueError(f"{source['dataset']} must use split='train'.")
        if source.get("data_files") != {"train": pattern}:
            raise ValueError(
                f"{source['dataset']} must use data_files={{'train': {pattern!r}}} "
                "to download the complete intended subset only."
            )


def download_one(
    source: dict[str, Any],
    sample_dir: Path,
    sample_rows: int,
    sample_max_chars: int,
) -> dict[str, Any]:
    output = _project_path(source.get("output"))
    if output is None:
        raise ValueError(f"source {source.get('source_id')} has no output path.")

    if source.get("source_type") == "manual" or source.get("format") == "manual":
        return check_manual_source(source)

    guard_dangerous_configs(source)

    if output.exists():
        try:
            dataset = load_local_dataset(output)
        except Exception as exc:
            print(
                f"Existing dataset is incomplete or unreadable: {output} "
                f"({type(exc).__name__}: {exc}); retrying download/save.",
                flush=True,
            )
        else:
            extra: dict[str, Any] = {"download_skipped": True}
            try:
                sample_path = write_dataset_sample(
                    source, dataset, sample_dir, sample_rows, sample_max_chars
                )
                extra["sample_path"] = str(sample_path.resolve())
                extra["summary"] = dataset_summary(dataset)
            except Exception as exc:
                extra["sample_error"] = f"{type(exc).__name__}: {exc}"
            return _source_record(source, "already_downloaded", output, extra)

    if source.get("source_type", "huggingface") != "huggingface":
        raise ValueError(
            f"Unsupported source_type for download: {source.get('source_type')}"
        )
    if source.get("format") != "disk":
        raise ValueError("download.py only supports format='disk'. Text export belongs to preprocess.py.")

    from datasets import DownloadConfig, enable_progress_bars, load_dataset
    from huggingface_hub.utils import enable_progress_bars as enable_hub_progress_bars

    enable_progress_bars()
    enable_hub_progress_bars()

    load_kwargs = {
        "path": source["dataset"],
        "name": source["config"],
        "split": source["split"],
        "revision": source["revision"],
        "data_files": source.get("data_files"),
    }
    print(f"Dataset: {source['dataset']}", flush=True)
    print(f"Split: {source['split']}", flush=True)
    if source.get("data_files") is not None:
        print(f"Files (all matching shards): {source['data_files']}", flush=True)
    print(f"Expected size: {source.get('size_note') or 'unknown'}", flush=True)
    print(f"Save to: {output.resolve()}", flush=True)
    print(f"Hugging Face cache: {DOWNLOAD_CACHE_ROOT}", flush=True)
    dataset = load_dataset(
        **{key: value for key, value in load_kwargs.items() if value is not None},
        cache_dir=str(DATASET_CACHE_DIR),
        download_config=DownloadConfig(
            cache_dir=str(RAW_DOWNLOAD_CACHE_DIR),
            download_desc=f"Downloading {source['dataset']}",
            disable_tqdm=False,
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))
    sample_path = write_dataset_sample(
        source, dataset, sample_dir, sample_rows, sample_max_chars
    )
    metadata = _source_record(
        source,
        "downloaded",
        output,
        {
            "sample_path": str(sample_path.resolve()),
            "summary": dataset_summary(dataset),
        },
    )
    metadata_path = output / "_download_meta.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Dataset saved to {output.resolve()}")
    print(f"Dataset sample written to {sample_path.resolve()}")
    return metadata


def main() -> None:
    config = Config(CONFIG_PATH)
    sources = config.require("downloads")
    manifest_path = _project_path(
        config.get("download_manifest", default=str(DEFAULT_MANIFEST_PATH))
    )
    sample_config = config.get("download_samples", default={})
    sample_dir = _project_path(sample_config.get("output_dir")) or DEFAULT_SAMPLE_DIR
    sample_rows = int(sample_config.get("rows", 2))
    sample_max_chars = int(sample_config.get("max_chars", 1000))
    cleanup_cache = bool(config.get("download_cache_cleanup", default=True))

    assert manifest_path is not None
    records: list[dict[str, Any]] = []

    try:
        for source in sources:
            if not source.get("enabled", False):
                records.append(_source_record(source, "disabled"))
                continue
            records.append(download_one(source, sample_dir, sample_rows, sample_max_chars))
    except Exception as exc:
        records.append(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_manifest(manifest_path, records)
        raise

    if not any(record["enabled"] for record in records):
        print("No enabled download sources in configs/data_pipeline.json")
    elif cleanup_cache:
        # 只有整轮跑完（没有抛异常）才清理：中途失败时保留缓存，重跑不必
        # 重新下载几十 GB。异常路径在下面的 except 里直接 re-raise，走不到这。
        cleanup_download_cache()
    else:
        print(f"Download cache kept at {DOWNLOAD_CACHE_ROOT.resolve()}")
    write_manifest(manifest_path, records)


if __name__ == "__main__":
    main()
