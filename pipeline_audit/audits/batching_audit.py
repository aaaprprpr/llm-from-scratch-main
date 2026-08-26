from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from tokenizers import Tokenizer as BackendTokenizer

from pipeline_audit.common import (
    AuditSection,
    resolve_project_path,
    tokenizer_file,
)
from data_pipeline import build_bin
from pretrain.train_model import TokenBatchLoader, get_batch


class _BackendBatchAdapter:
    """Expose the small transformers-like call used by build_bin.encode_batch."""

    def __init__(self, tokenizer: BackendTokenizer):
        self.backend = tokenizer

    def __call__(self, texts, **_kwargs):
        return {
            "input_ids": [
                self.backend.encode(text, add_special_tokens=False).ids
                for text in texts
            ]
        }


class _TinyDataset:
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return {"text": self.texts[item]}
        return {"text": self.texts[item]}


def _check_batch_exactness(section: AuditSection) -> None:
    data = np.arange(500, dtype=np.uint32)
    batch_size = 3
    context_length = 11
    position = 17
    tokens_per_batch = batch_size * context_length

    x, y, next_position = get_batch(
        data,
        batch_size,
        context_length,
        "cpu",
        position,
    )
    expected_x = torch.from_numpy(
        data[position : position + tokens_per_batch].astype(np.int64)
    ).view(batch_size, context_length)
    expected_y = torch.from_numpy(
        data[position + 1 : position + tokens_per_batch + 1].astype(np.int64)
    ).view(batch_size, context_length)

    exact = (
        torch.equal(x.cpu(), expected_x)
        and torch.equal(y.cpu(), expected_y)
        and next_position == position + tokens_per_batch
    )
    section.add(
        "get_batch_exact_shift",
        "pass" if exact else "fail",
        "x/y 与连续 token 流严格错开一个位置。" if exact else
        "get_batch 返回值与原始 token 流不一致。",
        position=position,
        next_position=next_position,
    )

    flattened_shift = torch.equal(x.flatten()[1:], y.flatten()[:-1])
    section.add(
        "flattened_next_token_alignment",
        "pass" if flattened_shift else "fail",
        "跨 batch 行边界的扁平 token 流仍保持 next-token 对齐。",
    )

    too_late = len(data) - (tokens_per_batch + 1) + 1
    wrapped_x, wrapped_y, wrapped_position = get_batch(
        data,
        batch_size,
        context_length,
        "cpu",
        too_late,
    )
    wrapped = (
        wrapped_x.flatten()[0].item() == 0
        and wrapped_y.flatten()[0].item() == 1
        and wrapped_position == tokens_per_batch
    )
    section.add(
        "end_of_file_wrap",
        "pass" if wrapped else "fail",
        "不足一个完整 batch 的文件尾会回绕到位置 0。",
        requested_position=too_late,
        returned_position=wrapped_position,
    )


def _collect_loader_batches(data, device: torch.device, count: int = 4):
    loader = TokenBatchLoader(
        data,
        batch_size=2,
        context_length=13,
        device=device,
        position=9,
    )
    batches = []
    for _ in range(count):
        x, y, position = loader.next()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batches.append((x.cpu().clone(), y.cpu().clone(), position))
    return batches


def _check_loader(section: AuditSection) -> None:
    data = np.arange(1000, dtype=np.uint32)
    cpu_batches = _collect_loader_batches(data, torch.device("cpu"))
    sequential = True
    expected_position = 9
    for x, y, returned_position in cpu_batches:
        expected_x, expected_y, expected_position = get_batch(
            data,
            2,
            13,
            "cpu",
            expected_position,
        )
        sequential &= (
            torch.equal(x, expected_x)
            and torch.equal(y, expected_y)
            and returned_position == expected_position
        )
    section.add(
        "cpu_loader_sequence",
        "pass" if sequential else "fail",
        "TokenBatchLoader 连续调用没有跳 batch 或重复 batch。",
    )

    resume_position = cpu_batches[1][2]
    resumed_loader = TokenBatchLoader(
        data,
        batch_size=2,
        context_length=13,
        device="cpu",
        position=resume_position,
    )
    resumed_x, resumed_y, resumed_next = resumed_loader.next()
    expected_x, expected_y, expected_next = get_batch(
        data,
        2,
        13,
        "cpu",
        resume_position,
    )
    resume_ok = (
        torch.equal(resumed_x, expected_x)
        and torch.equal(resumed_y, expected_y)
        and resumed_next == expected_next
    )
    section.add(
        "resume_position_continuity",
        "pass" if resume_ok else "fail",
        "保存的 train_position 能精确恢复到下一个尚未消费的 batch。",
        resume_position=resume_position,
    )

    if not torch.cuda.is_available():
        section.add(
            "cuda_prefetch_parity",
            "skip",
            "当前环境没有 CUDA；训练机运行时会比较异步预取与 CPU 路径。",
        )
        return

    device = torch.device("cuda")
    cuda_batches = _collect_loader_batches(data, device)
    parity = all(
        torch.equal(cpu_x, cuda_x)
        and torch.equal(cpu_y, cuda_y)
        and cpu_position == cuda_position
        for (cpu_x, cpu_y, cpu_position), (cuda_x, cuda_y, cuda_position)
        in zip(cpu_batches, cuda_batches, strict=True)
    )
    section.add(
        "cuda_prefetch_parity",
        "pass" if parity else "fail",
        "CUDA 双缓冲异步预取与 CPU 基准逐 token 一致。",
    )


def _check_bin_builder(section: AuditSection, config: dict) -> None:
    path = tokenizer_file(resolve_project_path(config["paths"]["tokenizer"]))
    tokenizer = BackendTokenizer.from_file(str(path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        section.add(
            "build_bin_eos_append",
            "fail",
            "Tokenizer 不包含 <|endoftext|>。",
        )
        return

    texts = ["第一篇短文。", "Second document 123.", "最后一篇。"]
    flags = [False, True, False]
    previous = (
        build_bin._worker_tokenizer,
        build_bin._worker_eos_id,
        build_bin._worker_dtype,
    )
    try:
        build_bin._worker_tokenizer = SimpleNamespace(
            tokenizer=_BackendBatchAdapter(tokenizer)
        )
        build_bin._worker_eos_id = eos_id
        build_bin._worker_dtype = np.dtype("uint16")
        train, validation, train_records, validation_records = (
            build_bin.encode_batch((texts, flags))
        )
    finally:
        (
            build_bin._worker_tokenizer,
            build_bin._worker_eos_id,
            build_bin._worker_dtype,
        ) = previous

    encoded = [
        tokenizer.encode(text, add_special_tokens=False).ids + [eos_id]
        for text in texts
    ]
    expected_train = np.asarray(encoded[0] + encoded[2], dtype=np.uint16)
    expected_validation = np.asarray(encoded[1], dtype=np.uint16)
    exact = (
        np.array_equal(train, expected_train)
        and np.array_equal(validation, expected_validation)
        and (train_records, validation_records) == (2, 1)
    )
    section.add(
        "build_bin_eos_and_split",
        "pass" if exact else "fail",
        "合成文档编码、单次 EOS 追加和 train/val 路由均与参考结果一致。",
        eos_id=eos_id,
    )

    dataset = _TinyDataset([f"record-{index}" for index in range(17)])
    emitted = list(
        build_bin.iter_shuffled_record_batches(
            dataset,
            train_ratio=0.8,
            seed=42,
            shuffle_block_records=5,
            batch_records=4,
        )
    )
    output_texts = [text for texts_, _ in emitted for text in texts_]
    output_flags = [flag for _, flags_ in emitted for flag in flags_]
    unique = len(output_texts) == len(set(output_texts)) == len(dataset)
    expected_train_count, expected_val_count = build_bin.split_sizes(
        len(dataset), 0.8
    )
    split_ok = (
        unique
        and output_flags.count(False) == expected_train_count
        and output_flags.count(True) == expected_val_count
    )
    section.add(
        "record_shuffle_and_split",
        "pass" if split_ok else "fail",
        "文档级打乱不会丢失、重复或跨 train/val 复制记录。",
    )


def run(config: dict, _report_dir) -> AuditSection:
    section = AuditSection("batching")
    _check_batch_exactness(section)
    _check_loader(section)
    _check_bin_builder(section, config)
    return section
