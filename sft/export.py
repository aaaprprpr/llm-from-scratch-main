import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from sft.utils import find_latest_checkpoint

LOGS_PATH = PROJECT_ROOT / "output" / "sft_logs"
OUTPUT_PATH = PROJECT_ROOT / "output" / "sft_weights" / "model.pt"


def main():
    checkpoint_path = find_latest_checkpoint(LOGS_PATH)
    print(f"加载 SFT checkpoint：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint["model"], OUTPUT_PATH)

    print(f"纯模型权重已保存到：{OUTPUT_PATH}")
    print(f"文件大小：{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
