from __future__ import annotations

import html
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer as BackendTokenizer

from pipeline_audit.common import (
    AuditSection,
    html_page,
    load_json,
    resolve_project_path,
    sha256_file,
    tokenizer_file,
)
from pretrain.train_model import get_batch


def _metadata_path(bin_path: Path) -> Path:
    return bin_path.with_suffix(bin_path.suffix + ".meta.json")


def _open_binary(bin_path: Path) -> tuple[np.memmap, dict]:
    metadata_path = _metadata_path(bin_path)
    metadata = load_json(metadata_path)
    dtype_name = metadata.get("dtype")
    if dtype_name not in {"uint16", "uint32"}:
        raise ValueError(f"Unsupported dtype {dtype_name!r} in {metadata_path}")
    return np.memmap(bin_path, dtype=np.dtype(dtype_name), mode="r"), metadata


def _sample_starts(
    total_tokens: int,
    span: int,
    count: int,
    rng: random.Random,
) -> list[int]:
    if total_tokens <= span:
        return [0]
    starts = {0, total_tokens - span}
    while len(starts) < min(count, total_tokens - span + 1):
        starts.add(rng.randrange(0, total_tokens - span + 1))
    return sorted(starts)


def _scan_token_range(
    data: np.memmap,
    vocab_size: int,
    audit_config: dict,
    rng: random.Random,
) -> dict:
    full_scan = audit_config["full_token_range_scan"]
    chunk_tokens = int(audit_config["full_scan_chunk_tokens"])
    sampled_regions = int(audit_config["sampled_token_regions"])
    region_tokens = min(int(audit_config["tokens_per_region"]), len(data))

    minimum = None
    maximum = None
    invalid = 0
    inspected = 0
    if full_scan:
        starts = range(0, len(data), chunk_tokens)
        spans = (
            np.asarray(data[start : start + chunk_tokens]) for start in starts
        )
        mode = "full"
    else:
        starts = _sample_starts(
            len(data),
            region_tokens,
            sampled_regions,
            rng,
        )
        spans = (
            np.asarray(data[start : start + region_tokens]) for start in starts
        )
        mode = "sampled"

    for values in spans:
        if not values.size:
            continue
        local_min = int(values.min())
        local_max = int(values.max())
        minimum = local_min if minimum is None else min(minimum, local_min)
        maximum = local_max if maximum is None else max(maximum, local_max)
        invalid += int(np.count_nonzero(values >= vocab_size))
        inspected += int(values.size)

    return {
        "mode": mode,
        "inspected_tokens": inspected,
        "minimum_token_id": minimum,
        "maximum_token_id": maximum,
        "out_of_vocabulary_tokens": invalid,
    }


def _decode_documents(
    data: np.memmap,
    tokenizer: BackendTokenizer,
    eos_id: int,
    count: int,
    radius: int,
    rng: random.Random,
) -> list[dict]:
    documents = []
    attempts = max(count * 8, 32)
    seen_ranges: set[tuple[int, int]] = set()

    for _ in range(attempts):
        if len(documents) >= count:
            break
        anchor = rng.randrange(0, len(data))
        region_start = max(0, anchor - radius)
        region_end = min(len(data), anchor + radius + 1)
        region = np.asarray(data[region_start:region_end])
        eos_offsets = np.flatnonzero(region == eos_id)
        local_anchor = anchor - region_start
        previous = eos_offsets[eos_offsets < local_anchor]
        following = eos_offsets[eos_offsets >= local_anchor]
        document_start = (
            region_start + int(previous[-1]) + 1 if previous.size else region_start
        )
        if not following.size:
            continue
        document_end = region_start + int(following[0])
        key = (document_start, document_end)
        if key in seen_ranges or document_end <= document_start:
            continue
        seen_ranges.add(key)

        token_ids = np.asarray(
            data[document_start:document_end],
            dtype=np.int64,
        ).tolist()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        documents.append(
            {
                "start": document_start,
                "end": document_end,
                "tokens": len(token_ids),
                "text": decoded,
                "truncated_left": document_start == region_start and region_start > 0,
            }
        )
    return documents


def _decode_batches(
    data: np.memmap,
    tokenizer: BackendTokenizer,
    eos_id: int,
    batch_size: int,
    sequence_length: int,
    count: int,
    rng: random.Random,
) -> list[dict]:
    tokens_per_batch = batch_size * sequence_length
    max_start = len(data) - tokens_per_batch - 1
    aligned_batches = max_start // tokens_per_batch
    positions = [0]
    while len(positions) < count and aligned_batches > 0:
        positions.append(rng.randrange(0, aligned_batches + 1) * tokens_per_batch)
        positions = list(dict.fromkeys(positions))

    batches = []
    for requested_position in positions:
        x, y, next_position = get_batch(
            data,
            batch_size,
            sequence_length,
            "cpu",
            requested_position,
        )
        raw_x = np.asarray(
            data[requested_position : requested_position + tokens_per_batch],
            dtype=np.int64,
        )
        raw_y = np.asarray(
            data[
                requested_position + 1 :
                requested_position + tokens_per_batch + 1
            ],
            dtype=np.int64,
        )
        exact = (
            np.array_equal(x.numpy().reshape(-1), raw_x)
            and np.array_equal(y.numpy().reshape(-1), raw_y)
        )
        rows = []
        for row_index in range(batch_size):
            row_ids = x[row_index].tolist()
            target_ids = y[row_index].tolist()
            absolute_start = requested_position + row_index * sequence_length
            rows.append(
                {
                    "row": row_index,
                    "absolute_start": absolute_start,
                    "starts_after_eos": (
                        absolute_start == 0
                        or int(data[absolute_start - 1]) == eos_id
                    ),
                    "eos_offsets": [
                        index for index, token_id in enumerate(row_ids)
                        if token_id == eos_id
                    ],
                    "text": tokenizer.decode(
                        row_ids,
                        skip_special_tokens=False,
                    ),
                    "target_text": tokenizer.decode(
                        target_ids,
                        skip_special_tokens=False,
                    ),
                    "first_token_pairs": [
                        [source, target]
                        for source, target in zip(
                            row_ids[:12],
                            target_ids[:12],
                            strict=True,
                        )
                    ],
                }
            )
        batches.append(
            {
                "position": requested_position,
                "next_position": next_position,
                "shift_exact": exact,
                "rows": rows,
            }
        )
    return batches


def _render_html(
    split_reports: list[dict],
    output_path: Path,
) -> None:
    body = []
    for split in split_reports:
        body.append(f"<h2>{html.escape(split['name'])}</h2>")
        body.append(
            "<p><code>"
            + html.escape(split["path"])
            + "</code></p>"
        )
        body.append("<h3>从 bin 恢复的文档</h3>")
        for index, document in enumerate(split["documents"], start=1):
            truncation = "（左侧可能被采样半径截断）" if document["truncated_left"] else ""
            body.append(
                f"<h4>文档 {index}: token [{document['start']}, "
                f"{document['end']}), {document['tokens']} tokens {truncation}</h4>"
            )
            body.append(f"<pre>{html.escape(document['text'])}</pre>")

        body.append("<h3>训练实际读取的 batch</h3>")
        for batch in split["batches"]:
            css = "ok" if batch["shift_exact"] else "fail"
            body.append(
                f"<h4>position={batch['position']}, "
                f"next={batch['next_position']}, "
                f"<span class='{css}'>shift_exact={batch['shift_exact']}</span></h4>"
            )
            for row in batch["rows"]:
                body.append(
                    f"<p>row={row['row']}, absolute_start={row['absolute_start']}, "
                    f"starts_after_EOS={row['starts_after_eos']}, "
                    f"EOS offsets={html.escape(str(row['eos_offsets']))}</p>"
                )
                body.append(
                    "<p>前 12 个 input→target token id：<code>"
                    + html.escape(str(row["first_token_pairs"]))
                    + "</code></p>"
                )
                body.append("<h5>input x</h5>")
                body.append(f"<pre>{html.escape(row['text'])}</pre>")
                body.append("<details><summary>展开对应 target y</summary>")
                body.append(f"<pre>{html.escape(row['target_text'])}</pre></details>")

    output_path.write_text(
        html_page("Token binary and training batch audit", "\n".join(body)),
        encoding="utf-8",
    )


def _audit_split(
    name: str,
    bin_path: Path,
    tokenizer: BackendTokenizer,
    tokenizer_path: Path,
    tokenizer_digest: str,
    vocab_size: int,
    eos_id: int,
    pretrain_config: dict,
    audit_config: dict,
    rng: random.Random,
    section: AuditSection,
) -> dict | None:
    metadata_path = _metadata_path(bin_path)
    if not bin_path.is_file():
        detail = f"缺少 {bin_path}；请在保存完整 bin 的训练机上运行。"
        if metadata_path.is_file():
            detail += " 当前只找到 metadata。"
        section.add(f"{name}_binary_available", "skip", detail)
        return None
    if not metadata_path.is_file():
        section.add(
            f"{name}_metadata_available",
            "fail",
            f"bin 存在但缺少 {metadata_path}。",
        )
        return None

    data, metadata = _open_binary(bin_path)
    dtype = np.dtype(metadata["dtype"])
    expected_bytes = int(metadata["tokens"]) * dtype.itemsize
    actual_bytes = bin_path.stat().st_size
    size_ok = expected_bytes == actual_bytes == len(data) * dtype.itemsize
    section.add(
        f"{name}_file_size_and_token_count",
        "pass" if size_ok else "fail",
        "metadata token 数、文件字节数和 memmap 长度一致。",
        metadata_tokens=int(metadata["tokens"]),
        memmap_tokens=len(data),
        expected_bytes=expected_bytes,
        actual_bytes=actual_bytes,
    )

    contract_ok = (
        metadata.get("tokenizer_size") == vocab_size
        and metadata.get("tokenizer_sha256") == tokenizer_digest
        and metadata.get("eos_id") == eos_id
    )
    section.add(
        f"{name}_tokenizer_contract",
        "pass" if contract_ok else "fail",
        "bin metadata 与当前 tokenizer 的词表、哈希和 EOS 一致。",
        metadata_tokenizer_size=metadata.get("tokenizer_size"),
        actual_tokenizer_size=vocab_size,
        metadata_tokenizer_sha256=metadata.get("tokenizer_sha256"),
        actual_tokenizer_sha256=tokenizer_digest,
        metadata_eos_id=metadata.get("eos_id"),
        actual_eos_id=eos_id,
        tokenizer=str(tokenizer_path),
    )

    range_result = _scan_token_range(
        data,
        vocab_size,
        audit_config,
        rng,
    )
    range_ok = range_result["out_of_vocabulary_tokens"] == 0
    section.add(
        f"{name}_token_id_range",
        "pass" if range_ok else "fail",
        f"{range_result['mode']} token 范围检查完成。",
        **range_result,
    )

    batch_size = int(pretrain_config["train"]["batch_size"])
    sequence_length = int(pretrain_config["train"]["sequence_length"])
    documents = _decode_documents(
        data,
        tokenizer,
        eos_id,
        int(audit_config["decoded_documents"]),
        int(audit_config["document_search_radius"]),
        rng,
    )
    batches = _decode_batches(
        data,
        tokenizer,
        eos_id,
        batch_size,
        sequence_length,
        int(audit_config["decoded_batches"]),
        rng,
    )
    shifts_ok = all(batch["shift_exact"] for batch in batches)
    section.add(
        f"{name}_real_batch_alignment",
        "pass" if shifts_ok else "fail",
        "真实 bin 经过生产 get_batch 后仍与原始连续 token 流精确对应。",
        decoded_batches=len(batches),
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    if len(documents) < int(audit_config["decoded_documents"]):
        section.add(
            f"{name}_document_recovery",
            "warn",
            "采样半径内没有恢复到配置数量的完整 EOS 边界文档。",
            requested=int(audit_config["decoded_documents"]),
            recovered=len(documents),
        )
    else:
        section.add(
            f"{name}_document_recovery",
            "pass",
            "已从 bin 中恢复配置数量的 EOS 分隔文档供人工查看。",
            recovered=len(documents),
        )

    return {
        "name": name,
        "path": str(bin_path),
        "metadata": metadata,
        "token_range": range_result,
        "documents": documents,
        "batches": batches,
    }


def run(config: dict, report_dir: Path) -> AuditSection:
    section = AuditSection("bin")
    tokenizer_path = tokenizer_file(
        resolve_project_path(config["paths"]["tokenizer"])
    )
    tokenizer = BackendTokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        section.add(
            "tokenizer_eos",
            "fail",
            "Tokenizer 没有 <|endoftext|>，无法解释文档边界。",
        )
        return section

    digest = sha256_file(tokenizer_path)
    pretrain_config = load_json(
        resolve_project_path(config["paths"]["pretrain_config"])
    )
    audit_config = config["bin_audit"]
    rng = random.Random(int(config["seed"]))
    split_reports = []

    for name, key in (("train", "train_bin"), ("validation", "val_bin")):
        result = _audit_split(
            name,
            resolve_project_path(config["paths"][key]),
            tokenizer,
            tokenizer_path,
            digest,
            vocab_size,
            eos_id,
            pretrain_config,
            audit_config,
            rng,
            section,
        )
        if result is not None:
            split_reports.append(result)

    model_context = int(pretrain_config["model"]["context_length"])
    train_context = int(pretrain_config["train"]["sequence_length"])
    context_status = "pass" if model_context == train_context else "warn"
    section.add(
        "trained_context_length",
        context_status,
        (
            "训练覆盖模型完整上下文长度。" if context_status == "pass" else
            "模型允许的上下文长于实际训练窗口；超出训练窗口的距离没有被直接训练。"
        ),
        model_context_length=model_context,
        training_sequence_length=train_context,
    )

    if split_reports:
        html_path = report_dir / "bin_samples.html"
        _render_html(split_reports, html_path)
        section.artifacts.append(str(html_path))
        section.metrics["splits"] = [
            {
                "name": item["name"],
                "tokens": item["metadata"].get("tokens"),
                "records": item["metadata"].get("records"),
                "decoded_documents": len(item["documents"]),
                "decoded_batches": len(item["batches"]),
            }
            for item in split_reports
        ]
    return section
