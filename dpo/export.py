import re
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_PATH = PROJECT_ROOT / "output" / "dpo_logs"
OUTPUT_PATH = PROJECT_ROOT / "output" / "dpo_weights" / "model.pt"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"ckpt_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint() -> Path:
    checkpoints = list(LOGS_PATH.glob("run_*/ckpt_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"没有找到 DPO checkpoint：{LOGS_PATH}")
    return max(
        checkpoints, key=lambda path: (checkpoint_step(path), path.stat().st_mtime)
    )


def main():
    checkpoint_path = find_latest_checkpoint()
    print(f"加载 DPO checkpoint：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint["model"], OUTPUT_PATH)

    print(f"纯模型权重已保存到：{OUTPUT_PATH}")
    print(f"文件大小：{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
