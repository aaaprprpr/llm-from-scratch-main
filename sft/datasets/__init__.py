from pathlib import Path

from sft.datasets.alpaca_zh import AlpacaZhDataset, load_alpaca_zh
from sft.datasets.coig_cqia import COIGCQIADataset, load_coig_cqia

DATASET_ADAPTERS = {
    "alpaca_zh": (load_alpaca_zh, AlpacaZhDataset),
    "coig_cqia": (load_coig_cqia, COIGCQIADataset),
}


def load_chat_dataset(adapter: str, path: str | Path):
    try:
        loader, dataset_class = DATASET_ADAPTERS[adapter]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_ADAPTERS))
        raise ValueError(
            f"不支持的 SFT 数据集适配器 {adapter!r}；可选：{supported}"
        ) from exc
    return dataset_class(loader(path))
