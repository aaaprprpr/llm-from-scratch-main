from __future__ import annotations

import csv
import gc
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as BackendTokenizer

from pipeline_audit.common import (
    AuditSection,
    load_json,
    resolve_project_path,
    tokenizer_file,
)
from models.model import Transformer


def _autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _position_buckets(sequence_length: int) -> list[tuple[int, int]]:
    boundaries = [0, 64, 256, 512, 1024, sequence_length]
    boundaries = sorted({min(max(value, 0), sequence_length) for value in boundaries})
    return [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]


def _forward_token_losses(
    model: Transformer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), _autocast(device):
        logits, _ = model(x, use_cache=False)
    losses = F.cross_entropy(
        logits[0].float(),
        y[0],
        reduction="none",
    )
    return logits, losses


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_metrics(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "category",
        "window_index",
        "absolute_start",
        "metric",
        "value",
        "position_start",
        "position_end",
        "context_length",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict, report_dir: Path) -> AuditSection:
    section = AuditSection("checkpoint")
    audit_config = config["checkpoint_audit"]
    if not audit_config.get("enabled", True):
        section.add("checkpoint_audit_enabled", "skip", "配置中已关闭。")
        return section

    checkpoint_value = config["paths"].get("checkpoint")
    if not checkpoint_value:
        section.add(
            "checkpoint_available",
            "skip",
            "尚未配置本机 checkpoint；设置 paths.checkpoint 后运行真实 checkpoint 审计。",
        )
        return section
    checkpoint_path = resolve_project_path(checkpoint_value)
    train_bin_path = resolve_project_path(config["paths"]["train_bin"])
    metadata_path = train_bin_path.with_suffix(train_bin_path.suffix + ".meta.json")
    if not checkpoint_path.is_file():
        section.add(
            "checkpoint_available",
            "skip",
            f"缺少 checkpoint: {checkpoint_path}",
        )
        return section
    if not train_bin_path.is_file():
        section.add(
            "checkpoint_real_batch_probe",
            "skip",
            f"缺少真实 train.bin: {train_bin_path}；不加载大 checkpoint。",
        )
        return section
    if not metadata_path.is_file():
        section.add(
            "checkpoint_real_batch_probe",
            "fail",
            f"缺少 train.bin metadata: {metadata_path}",
        )
        return section

    pretrain_config = load_json(
        resolve_project_path(config["paths"]["pretrain_config"])
    )
    metadata = load_json(metadata_path)
    dtype_name = metadata.get("dtype")
    if dtype_name not in {"uint16", "uint32"}:
        section.add(
            "train_bin_dtype",
            "fail",
            f"不支持的 dtype: {dtype_name!r}",
        )
        return section
    data = np.memmap(train_bin_path, dtype=np.dtype(dtype_name), mode="r")

    tokenizer_path = tokenizer_file(
        resolve_project_path(config["paths"]["tokenizer"])
    )
    tokenizer = BackendTokenizer.from_file(str(tokenizer_path))
    tokenizer_size = tokenizer.get_vocab_size(with_added_tokens=True)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model_args = dict(checkpoint["model_args"])
    configured_model_args = dict(pretrain_config["model"])
    args_match = model_args == configured_model_args
    section.add(
        "checkpoint_model_config",
        "pass" if args_match else "warn",
        "checkpoint model_args 与当前预训练配置一致。" if args_match else
        "checkpoint model_args 与当前配置不同；报告使用 checkpoint 自带配置。",
        checkpoint_model_args=model_args,
        configured_model_args=configured_model_args,
    )
    vocab_ok = model_args["vocab_size"] == tokenizer_size
    section.add(
        "checkpoint_tokenizer_size",
        "pass" if vocab_ok else "fail",
        "checkpoint 词表大小与 tokenizer 一致。",
        checkpoint_vocab_size=model_args["vocab_size"],
        tokenizer_size=tokenizer_size,
    )
    if not vocab_ok:
        del checkpoint
        return section

    model = Transformer(**model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    iteration = int(checkpoint["iteration"])
    tokens_seen = checkpoint.get("tokens_seen")
    del checkpoint

    tied_expected = bool(model_args.get("tie_word_embeddings", False))
    tied_actual = model.embedding.weight is model.lm_head.weight
    section.add(
        "checkpoint_weight_tying",
        "pass" if tied_expected == tied_actual else "fail",
        "加载 checkpoint 后 embedding/lm_head 权重共享状态正确。",
        expected=tied_expected,
        actual=tied_actual,
    )

    nonfinite_parameters = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            nonfinite_parameters.append(name)
    section.add(
        "checkpoint_parameters_finite",
        "pass" if not nonfinite_parameters else "fail",
        "全部 checkpoint 参数均为有限值。",
        nonfinite_parameters=nonfinite_parameters,
    )

    device = _resolve_device(str(audit_config.get("device", "auto")))
    model.to(device).eval()
    sequence_length = int(pretrain_config["train"]["sequence_length"])
    if sequence_length > model_args["context_length"]:
        section.add(
            "checkpoint_sequence_length",
            "fail",
            "训练 sequence_length 超过 checkpoint context_length。",
        )
        return section

    sampled_windows = int(audit_config["sampled_windows"])
    rng = random.Random(int(config["seed"]) ^ iteration)
    max_start = len(data) - sequence_length - 1
    if max_start < 0:
        section.add(
            "checkpoint_real_batch_probe",
            "fail",
            "train.bin 不足一个完整 sequence。",
        )
        return section
    starts = [0]
    while len(starts) < sampled_windows:
        starts.append(rng.randrange(0, max_start + 1))
        starts = list(dict.fromkeys(starts))

    rows: list[dict] = []
    correct_losses = []
    wrong_shift_losses = []
    shuffled_losses = []
    eos_losses = []
    post_eos_losses = []
    first_x = None
    first_y = None

    for window_index, start in enumerate(starts):
        window = np.asarray(
            data[start : start + sequence_length + 1],
            dtype=np.int64,
        )
        x = torch.from_numpy(window[:-1].copy())[None, :].to(device)
        y = torch.from_numpy(window[1:].copy())[None, :].to(device)
        logits, token_losses = _forward_token_losses(model, x, y, device)
        correct_loss = token_losses.mean().item()

        wrong_y = y.roll(shifts=-1, dims=1)
        wrong_shift_loss = F.cross_entropy(
            logits[0].float(),
            wrong_y[0],
        ).item()
        permutation = torch.randperm(
            sequence_length,
            generator=torch.Generator().manual_seed(int(config["seed"]) + window_index),
        ).to(device)
        shuffled_loss = F.cross_entropy(
            logits[0].float(),
            y[0, permutation],
        ).item()

        correct_losses.append(correct_loss)
        wrong_shift_losses.append(wrong_shift_loss)
        shuffled_losses.append(shuffled_loss)
        rows.extend(
            [
                {
                    "category": "window",
                    "window_index": window_index,
                    "absolute_start": start,
                    "metric": "correct_next_token_loss",
                    "value": correct_loss,
                    "position_start": 0,
                    "position_end": sequence_length,
                    "context_length": sequence_length,
                },
                {
                    "category": "window",
                    "window_index": window_index,
                    "absolute_start": start,
                    "metric": "wrong_shift_loss",
                    "value": wrong_shift_loss,
                    "position_start": 0,
                    "position_end": sequence_length,
                    "context_length": sequence_length,
                },
                {
                    "category": "window",
                    "window_index": window_index,
                    "absolute_start": start,
                    "metric": "shuffled_label_loss",
                    "value": shuffled_loss,
                    "position_start": 0,
                    "position_end": sequence_length,
                    "context_length": sequence_length,
                },
            ]
        )

        for position_start, position_end in _position_buckets(sequence_length):
            bucket_loss = token_losses[position_start:position_end].mean().item()
            rows.append(
                {
                    "category": "position_bucket",
                    "window_index": window_index,
                    "absolute_start": start,
                    "metric": "correct_next_token_loss",
                    "value": bucket_loss,
                    "position_start": position_start,
                    "position_end": position_end,
                    "context_length": sequence_length,
                }
            )

        if eos_id is not None:
            eos_mask = y[0].eq(eos_id)
            post_eos_mask = x[0].eq(eos_id)
            if eos_mask.any():
                value = token_losses[eos_mask].mean().item()
                eos_losses.append(value)
                rows.append(
                    {
                        "category": "boundary",
                        "window_index": window_index,
                        "absolute_start": start,
                        "metric": "eos_target_loss",
                        "value": value,
                        "position_start": "",
                        "position_end": "",
                        "context_length": sequence_length,
                    }
                )
            if post_eos_mask.any():
                value = token_losses[post_eos_mask].mean().item()
                post_eos_losses.append(value)
                rows.append(
                    {
                        "category": "boundary",
                        "window_index": window_index,
                        "absolute_start": start,
                        "metric": "first_token_after_eos_loss",
                        "value": value,
                        "position_start": "",
                        "position_end": "",
                        "context_length": sequence_length,
                    }
                )

        if first_x is None:
            first_x, first_y = x, y
        del logits, token_losses

    mean_correct = _mean_or_none(correct_losses)
    mean_wrong = _mean_or_none(wrong_shift_losses)
    mean_shuffled = _mean_or_none(shuffled_losses)
    label_order_ok = (
        mean_correct is not None
        and mean_wrong is not None
        and mean_shuffled is not None
        and mean_correct < mean_wrong
        and mean_correct < mean_shuffled
    )
    section.add(
        "trained_label_alignment",
        "pass" if label_order_ok else "warn",
        "正确 next-token 标签的 loss 低于错位和乱序标签。" if label_order_ok else
        "正确标签没有明显优于错误标签，需要检查标签或 checkpoint 学习状态。",
        correct_loss=mean_correct,
        wrong_shift_loss=mean_wrong,
        shuffled_label_loss=mean_shuffled,
    )

    scored_tokens = int(audit_config["scored_tail_tokens"])
    context_lengths = sorted(
        {
            int(value)
            for value in audit_config["context_lengths"]
            if scored_tokens < int(value) <= sequence_length
        }
    )
    context_curve = []
    assert first_x is not None and first_y is not None
    target_tail = first_y[:, -scored_tokens:]
    for context_length in context_lengths:
        context_x = first_x[:, -context_length:]
        with torch.inference_mode(), _autocast(device):
            context_logits, _ = model(context_x, use_cache=False)
        value = F.cross_entropy(
            context_logits[0, -scored_tokens:].float(),
            target_tail[0],
        ).item()
        context_curve.append(
            {"context_length": context_length, "loss": value}
        )
        rows.append(
            {
                "category": "context_ablation",
                "window_index": 0,
                "absolute_start": starts[0],
                "metric": "same_tail_loss",
                "value": value,
                "position_start": sequence_length - scored_tokens,
                "position_end": sequence_length,
                "context_length": context_length,
            }
        )
        del context_logits

    if context_curve:
        shortest = context_curve[0]["loss"]
        longest = context_curve[-1]["loss"]
        context_status = "pass" if longest <= shortest + 0.05 else "warn"
        section.add(
            "trained_context_ablation",
            context_status,
            "同一批尾部目标在不同前文长度下的 loss 已记录；长上下文未明显恶化。"
            if context_status == "pass" else
            "同一目标使用最长上下文反而明显变差，需要扩大采样确认。",
            scored_tail_tokens=scored_tokens,
            curve=context_curve,
        )
    else:
        section.add(
            "trained_context_ablation",
            "skip",
            "没有合法的 context_lengths；每项必须大于 scored_tail_tokens 且不超过 sequence_length。",
        )

    metrics_path = report_dir / "checkpoint_metrics.csv"
    _write_metrics(metrics_path, rows)
    section.artifacts.append(str(metrics_path))
    section.metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "iteration": iteration,
            "tokens_seen": tokens_seen,
            "device": str(device),
            "sampled_windows": len(starts),
            "sequence_length": sequence_length,
            "mean_correct_loss": mean_correct,
            "mean_wrong_shift_loss": mean_wrong,
            "mean_shuffled_label_loss": mean_shuffled,
            "mean_eos_target_loss": _mean_or_none(eos_losses),
            "mean_first_token_after_eos_loss": _mean_or_none(post_eos_losses),
            "context_curve": context_curve,
        }
    )

    del model, first_x, first_y
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return section
