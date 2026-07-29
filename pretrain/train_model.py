import torch
import numpy as np
import math
import os
from typing import BinaryIO, IO


def lr_cosine_schedule(
    step,
    max_lr,
    min_lr,
    warmup_steps,
    decay_end_step,
):
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

    if decay_end_step <= warmup_steps:
        raise ValueError("decay_end_step must be greater than warmup_steps")

    if not 0 <= min_lr <= max_lr:
        raise ValueError(f"Expected 0 <= min_lr <= max_lr, got {min_lr}, {max_lr}")

    # step 从 0 开始，第一步使用非零学习率
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step >= decay_end_step:
        return min_lr

    decay_ratio = (step - warmup_steps) / (decay_end_step - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + cosine * (max_lr - min_lr)


def get_batch(
    data,
    batch_size,
    context_length,
    device,
    position,
):
    device = torch.device(device)

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    if len(data) <= context_length:
        raise ValueError(
            f"data must contain at least {context_length + 1} tokens, "
            f"got {len(data)}"
        )

    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")

    # 1. 计算总共有多少个合法的起始位置
    max_start = len(data) - context_length - 1

    # 2. 生成 batch_size 个索引
    # 不再是 randint，而是从当前位置开始往后排
    # 比如：[pos, pos + context_length, pos + 2*context_length, ...]
    # 这样能保证数据被地毯式扫过
    indices = []

    for _ in range(batch_size):
        if position > max_start:
            position = 0  # 扫完了，回到开头
        indices.append(position)
        position += context_length  # 步长等于上下文长度，无缝衔接

    # 每条序列只读取和转换一次 T+1 token，再在目标设备上切出 x/y。
    # 旧实现会分别为 x 和 y 重复 memmap 切片、astype、stack 和设备搬运。
    windows = np.stack(
        [data[i : i + context_length + 1] for i in indices]
    ).astype(np.int64, copy=False)
    tokens = torch.from_numpy(windows)
    tokens = tokens.to(device)

    return tokens[:, :-1], tokens[:, 1:], position


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    train_position=None,
    tokens_seen=None,
    model_args=None,
    config=None,
):
    """
    将前三个参数的所有状态转储到类文件对象 out 中。
    """

    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if train_position is not None:
        obj["train_position"] = train_position
    if tokens_seen is not None:
        obj["tokens_seen"] = tokens_seen
    if model_args is not None:
        obj["model_args"] = model_args
    if config is not None:
        obj["config"] = config
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location="cpu",
):
    obj = torch.load(
        src,
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return (
        obj["iteration"],
        obj.get("train_position", 0),
        obj.get("tokens_seen"),
    )
