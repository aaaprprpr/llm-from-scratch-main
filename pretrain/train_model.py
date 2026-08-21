import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO, IO

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def tokenizer_fingerprint(path: str | os.PathLike) -> str:
    path = Path(path)
    target = path / "tokenizer.json" if path.is_dir() else path
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_token_bin(
    path: str | os.PathLike,
    *,
    expected_tokenizer_size: int | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> np.memmap:
    path = Path(path)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dtype_name = metadata.get("dtype")
    if dtype_name not in {"uint16", "uint32"}:
        raise ValueError(
            f"{metadata_path} contains unsupported dtype {dtype_name!r}"
        )
    if (
        expected_tokenizer_size is not None
        and metadata.get("tokenizer_size") != expected_tokenizer_size
    ):
        raise ValueError(
            f"{metadata_path} was built with tokenizer size "
            f"{metadata.get('tokenizer_size')!r}, but pretraining expects "
            f"{expected_tokenizer_size}. Rebuild the token binaries."
        )
    if (
        expected_tokenizer_sha256 is not None
        and metadata.get("tokenizer_sha256") != expected_tokenizer_sha256
    ):
        raise ValueError(
            f"{metadata_path} was built with a different tokenizer "
            "fingerprint. Rebuild the token binaries."
        )
    return np.memmap(path, dtype=np.dtype(dtype_name), mode="r")


def get_device(config) -> torch.device:
    device_name = config.get("train", "device", default="auto")
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_dtype(precision: str, device: torch.device):
    if precision == "float32":
        return None
    if precision != "bfloat16":
        raise ValueError(
            f"train.precision must be 'float32' or 'bfloat16', got {precision!r}"
        )
    if device.type != "cuda":
        return None
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "当前 CUDA 设备不支持 bfloat16；请把 train.precision 改为 float32"
        )
    return torch.bfloat16


def autocast_context(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def attention_kernel_context(device: torch.device, require_flash: bool):
    if device.type == "cuda" and require_flash:
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    return nullcontext()


def verify_flash_attention(device: torch.device, dtype) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            "train.require_flash_attention=true，但当前训练设备不是 CUDA"
        )
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashAttention baseline requires float16 or bfloat16")

    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        1,
        8,
        128,
        64,
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = F.scaled_dot_product_attention(q, q, q, is_causal=True)
            output.square().mean().backward()
            torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(
            "当前设备或 PyTorch 构建无法运行 FlashAttention SDPA"
        ) from exc
    print("FlashAttention SDPA forward/backward verification passed")


def resolve_training_parameters(model_config: dict, train_config: dict):
    sequence_length = train_config.get(
        "sequence_length",
        model_config["context_length"],
    )
    if sequence_length <= 0:
        raise ValueError(
            f"train.sequence_length must be positive, got {sequence_length}"
        )
    if sequence_length > model_config["context_length"]:
        raise ValueError(
            f"train.sequence_length {sequence_length} exceeds model.context_length "
            f"{model_config['context_length']}"
        )

    batch_size = train_config["batch_size"]
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(
            f"train.batch_size must be a positive integer, got {batch_size}"
        )
    tokens_per_micro_batch = batch_size * sequence_length

    configured_tokens_per_update = train_config.get("tokens_per_update")
    if configured_tokens_per_update is not None:
        if configured_tokens_per_update < tokens_per_micro_batch:
            raise ValueError(
                "train.tokens_per_update cannot be smaller than one micro-batch "
                f"({tokens_per_micro_batch})"
            )
        if configured_tokens_per_update % tokens_per_micro_batch != 0:
            raise ValueError(
                "train.tokens_per_update must be divisible by "
                f"batch_size * sequence_length ({tokens_per_micro_batch})"
            )
        gradient_accumulation_steps = (
            configured_tokens_per_update // tokens_per_micro_batch
        )
    else:
        gradient_accumulation_steps = train_config.get(
            "gradient_accumulation_steps",
            1,
        )
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "train.gradient_accumulation_steps must be positive"
            )
    tokens_per_update = tokens_per_micro_batch * gradient_accumulation_steps

    eval_tokens = train_config.get("eval_tokens")
    if eval_tokens is None:
        eval_iters = train_config["eval_iters"]
    else:
        if eval_tokens <= 0:
            raise ValueError(
                f"train.eval_tokens must be positive, got {eval_tokens}"
            )
        eval_iters = max(
            1,
            (eval_tokens + tokens_per_micro_batch - 1)
            // tokens_per_micro_batch,
        )

    return (
        sequence_length,
        batch_size,
        tokens_per_micro_batch,
        gradient_accumulation_steps,
        tokens_per_update,
        eval_iters,
    )


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


@torch.inference_mode()
def estimate_loss(
    model,
    data,
    batch_size,
    context_length,
    device,
    eval_iters,
    amp_dtype=None,
    require_flash_attention=False,
    start_position=0,
):
    """在固定数据窗口上计算平均 loss，不修改训练游标。"""
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    was_training = model.training
    total_loss = torch.zeros((), device=device)
    batches = TokenBatchLoader(
        data,
        batch_size,
        context_length,
        device,
        position=start_position,
    )

    model.eval()
    try:
        for _ in range(eval_iters):
            x, y, _ = batches.next()
            with attention_kernel_context(device, require_flash_attention):
                with autocast_context(device, amp_dtype):
                    logits, _ = model(x, use_cache=False)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    )
            total_loss += loss
        return (total_loss / eval_iters).item()
    finally:
        model.train(was_training)


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
