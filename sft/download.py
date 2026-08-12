import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import concatenate_datasets, load_dataset, load_from_disk

from sft.datasets.instruction import REQUIRED_COLUMNS
from sft.utils import load_config


def validate_columns(dataset, dataset_name: str) -> None:
    missing_columns = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise KeyError(f"{dataset_name} 缺少字段：{names}")


def download_dataset(config) -> None:
    data_config = config.require("data")
    dataset_id = data_config["dataset_id"]
    split = data_config.get("split", "train")
    revision = data_config.get("revision")
    subset_names = data_config.get("subsets") or [None]
    output = config.resolve_path("paths", "dataset")

    if output.exists():
        dataset = load_from_disk(str(output))
        validate_columns(dataset, dataset_id)
        print(f"SFT 数据集已存在：{output}，共 {len(dataset)} 条")
        return

    subsets = []
    for subset_name in subset_names:
        label = subset_name or "default"
        print(f"下载 {dataset_id} / {label} / {split}")
        kwargs = {
            "path": dataset_id,
            "name": subset_name,
            "split": split,
            "revision": revision,
        }
        dataset = load_dataset(
            **{key: value for key, value in kwargs.items() if value is not None}
        )
        validate_columns(dataset, f"{dataset_id}/{label}")
        subsets.append(dataset.select_columns(sorted(REQUIRED_COLUMNS)))
        print(f"已读取 {len(dataset)} 条")

    dataset = subsets[0] if len(subsets) == 1 else concatenate_datasets(subsets)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))
    print(f"SFT 数据集已保存：{output}，共 {len(dataset)} 条")


def main() -> None:
    download_dataset(load_config())


if __name__ == "__main__":
    main()
