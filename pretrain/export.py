import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from checkpoint_io import load_checkpoint

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "output"
    / "train_logs"
    / "run_20260706_175441"
    / "ckpt_step_130000.pt"
)
OUTPUT_PATH = PROJECT_ROOT / "output" / "pretrained_weights" / "model.pt"


def main():
    checkpoint = load_checkpoint(CHECKPOINT_PATH, map_location="cpu", mmap=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint["model"], OUTPUT_PATH)

    print(f"纯模型权重已保存到：{OUTPUT_PATH}")
    print(f"文件大小：{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
