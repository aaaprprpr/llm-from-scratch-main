import csv
import time
from pathlib import Path

import torch


def print_config(config_data: dict) -> None:
    print("=" * 20 + " Training Configurations " + "=" * 20)
    for section, values in config_data.items():
        print(f"[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"{key:20}: {value}")
        else:
            print(values)
    print("=" * 65)


class TrainingTracker:
    def __init__(
        self,
        run_dir: Path,
        device: torch.device,
        start_step: int = 0,
        train_position: int = 0,
        tokens_seen: int = 0,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_config: dict | None = None,
    ):
        self.device = device
        self.train_position = train_position
        self.tokens_seen = tokens_seen
        self.last_log_step = start_step
        self.last_log_tokens = tokens_seen
        self.cuda_step_events = []
        self.accumulated_training_time = 0.0
        self.step_start = None

        self.metrics_path = run_dir / "metrics.csv"
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    ["step", "tokens_seen", "train_loss", "val_loss", "lr"]
                )

        self.train_metrics_path = run_dir / "train_metrics.csv"
        if not self.train_metrics_path.exists():
            with self.train_metrics_path.open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                csv.writer(stream).writerow(
                    [
                        "step",
                        "tokens_seen",
                        "train_position",
                        "loss",
                        "lr",
                        "grad_norm",
                        "ms_per_step",
                        "tokens_per_second",
                    ]
                )

        self.samples_path = run_dir / "samples.csv"
        if not self.samples_path.exists():
            with self.samples_path.open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                csv.writer(stream).writerow(
                    [
                        "step",
                        "tokens_seen",
                        "prompt",
                        "completion",
                        "full_text",
                        "temperature",
                        "top_p",
                        "max_new_tokens",
                    ]
                )

        self.wandb = None
        if use_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError(
                    "logging.use_wandb=true，但当前环境没有安装 wandb"
                ) from exc
            wandb.init(project=wandb_project, config=wandb_config)
            self.wandb = wandb

    def start_training_step(self) -> None:
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.step_start = (start, end)
        else:
            self.step_start = time.perf_counter()

    def end_training_step(self) -> None:
        if self.device.type == "cuda":
            start, end = self.step_start
            end.record()
            self.cuda_step_events.append((start, end))
        else:
            self.accumulated_training_time += (
                time.perf_counter() - self.step_start
            )

    def record_batch(self, train_position: int, token_count: int) -> None:
        self.train_position = train_position
        self.tokens_seen += token_count

    def log_training_step(
        self,
        step: int,
        loss: torch.Tensor,
        grad_norm: torch.Tensor,
        lr: float,
    ) -> None:
        if self.device.type == "cuda":
            self.cuda_step_events[-1][1].synchronize()
            training_time = sum(
                start.elapsed_time(end) for start, end in self.cuda_step_events
            ) / 1000.0
        else:
            training_time = self.accumulated_training_time

        steps = step - self.last_log_step
        ms_per_step = training_time * 1000 / steps
        tokens_per_second = (
            self.tokens_seen - self.last_log_tokens
        ) / training_time
        loss_value = loss.item()
        grad_norm_value = grad_norm.item()
        print(
            f"step {step}: "
            f"loss {loss_value:.4f}, "
            f"lr {lr:.2e}, "
            f"time {ms_per_step:.2f}ms/step, "
            f"tokens/s {tokens_per_second:.0f}, "
            f"grad_norm {grad_norm_value:.4f}"
        )

        with self.train_metrics_path.open(
            "a", newline="", encoding="utf-8"
        ) as stream:
            csv.writer(stream).writerow(
                [
                    step,
                    self.tokens_seen,
                    self.train_position,
                    loss_value,
                    lr,
                    grad_norm_value,
                    ms_per_step,
                    tokens_per_second,
                ]
            )

        if self.wandb is not None:
            self.wandb.log(
                {
                    "step": step,
                    "tokens_seen": self.tokens_seen,
                    "train/step_loss": loss_value,
                    "train/grad_norm": grad_norm_value,
                    "train/tokens_per_second": tokens_per_second,
                    "lr": lr,
                }
            )

        self.last_log_step = step
        self.last_log_tokens = self.tokens_seen
        self.cuda_step_events.clear()
        self.accumulated_training_time = 0.0

    def log_evaluation(
        self,
        step: int,
        train_loss: float,
        val_loss: float,
        lr: float,
    ) -> None:
        print(
            f"Step {step}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, "
            f"lr {lr:.2e}"
        )

        if self.wandb is not None:
            self.wandb.log(
                {
                    "step": step,
                    "tokens_seen": self.tokens_seen,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": lr,
                }
            )

        with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [step, self.tokens_seen, train_loss, val_loss, lr]
            )

    def log_sample(
        self,
        step: int,
        prompt: str,
        completion: str,
        full_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> None:
        with self.samples_path.open(
            "a", newline="", encoding="utf-8"
        ) as stream:
            csv.writer(stream).writerow(
                [
                    step,
                    self.tokens_seen,
                    prompt,
                    completion,
                    full_text,
                    temperature,
                    top_p,
                    max_new_tokens,
                ]
            )
