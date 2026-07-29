import json
import math
import os
from pathlib import Path
from typing import BinaryIO, IO

import torch
import numpy as np


def load_token_bin(path: str | os.PathLike) -> np.memmap:
    path = Path(path)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dtype_name = metadata.get("dtype")
    if dtype_name not in {"uint16", "uint32"}:
        raise ValueError(
            f"{metadata_path} contains unsupported dtype {dtype_name!r}"
        )
    return np.memmap(path, dtype=np.dtype(dtype_name), mode="r")


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

    tokens_per_batch = batch_size * context_length
    window_length = tokens_per_batch + 1
    if len(data) < window_length:
        raise ValueError(
            f"data must contain at least {window_length} tokens for a full batch, "
            f"got {len(data)}"
        )

    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")

    max_batch_start = len(data) - window_length
    if position > max_batch_start:
        position = 0

    window = np.asarray(
        data[position : position + window_length],
        dtype=np.int64,
    )
    tokens = torch.from_numpy(window).to(device)
    next_position = position + tokens_per_batch

    x = tokens[:-1].view(batch_size, context_length)
    y = tokens[1:].view(batch_size, context_length)
    return x, y, next_position


class TokenBatchLoader:
    def __init__(
        self,
        data,
        batch_size: int,
        context_length: int,
        device,
        position: int = 0,
    ):
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = torch.device(device)
        self.position = position
        self.tokens_per_batch = batch_size * context_length
        self.window_length = self.tokens_per_batch + 1

        if self.device.type != "cuda":
            self.copy_stream = None
            return

        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.host_buffers = [
            torch.empty(
                self.window_length,
                dtype=torch.long,
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self.copy_events = [None, None]
        self.ready_batch = None
        self._preload(slot=0)

    def _preload(self, slot: int) -> None:
        previous_copy = self.copy_events[slot]
        if previous_copy is not None:
            previous_copy.synchronize()

        max_batch_start = len(self.data) - self.window_length
        if self.position > max_batch_start:
            self.position = 0

        np.copyto(
            self.host_buffers[slot].numpy(),
            self.data[self.position : self.position + self.window_length],
            casting="safe",
        )
        next_position = self.position + self.tokens_per_batch

        with torch.cuda.stream(self.copy_stream):
            device_window = self.host_buffers[slot].to(
                self.device,
                non_blocking=True,
            )
            copy_done = torch.cuda.Event()
            copy_done.record(self.copy_stream)

        self.copy_events[slot] = copy_done
        self.ready_batch = (slot, device_window, copy_done, next_position)
        self.position = next_position

    def next(self):
        if self.copy_stream is None:
            x, y, self.position = get_batch(
                self.data,
                self.batch_size,
                self.context_length,
                self.device,
                self.position,
            )
            return x, y, self.position

        slot, device_window, copy_done, next_position = self.ready_batch
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(copy_done)
        device_window.record_stream(current_stream)

        self._preload(slot=1 - slot)

        x = device_window[:-1].view(self.batch_size, self.context_length)
        y = device_window[1:].view(self.batch_size, self.context_length)
        return x, y, next_position


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
