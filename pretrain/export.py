from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "output"
    / "train_logs"
    / "run_20260706_175441"
    / "ckpt_step_130000.pt"
)
OUTPUT_PATH = PROJECT_ROOT / "output" / "pretrained_weights" / "model.pt"


def main():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
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
