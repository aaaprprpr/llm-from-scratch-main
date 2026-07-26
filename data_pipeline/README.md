# 数据流水线

数据相关代码统一从 `configs/data_pipeline.json` 读取配置，不再传命令行参数。所有命令都从仓库根目录执行。

整体流程：

```text
download.py    -> 下载/管理原始 dataset，并生成结构样本
preprocess.py  -> 从已下载 dataset 读取，adapter 格式化，清洗并切 train/val
build_bin.py   -> 从清洗后的 train/val 文本生成 token bin
```

## 1. 下载数据集

配置位置：`configs/data_pipeline.json` 的 `downloads`。

`download.py` 只负责三件事：

- 按 `downloads` 里的启用项下载 Hugging Face dataset；
- 如果目标目录已经存在，直接标记为 `already_downloaded`，不会重新触发下载；
- 在 `data_pipeline/dataset_samples/` 下生成 `<source_id>.sample.json`，方便查看字段结构和少量样本。

运行：

```powershell
python .\data_pipeline\download.py
```

注意：

- Hugging Face 数据源统一使用 `format: "disk"`，下载为 `data/downloads/<source_id>`。
- `download.py` 不再导出 txt。
- `wikimedia/wikipedia` 必须显式写 `config: "20231101.zh"` 和 `split: "train"`。不要把 wiki 配成 `config: null`，避免误拉全量 wikipedia。

## 2. 数据预处理

配置位置：`configs/data_pipeline.json` 的 `preprocess`。

`preprocess.py` 负责：

- 读取 `preprocess.dataset_sources` 中启用的数据源；
- 从对应的 `data/downloads/...` 目录加载 dataset；
- 按每个数据源的 `adapter` 转成一条一条预训练文本；
- 清洗文本；
- 按固定随机种子切分 train/val；
- 输出 `data/train.txt`、`data/val.txt` 和 `data/preprocess.meta.json`。

运行：

```powershell
python .\data_pipeline\preprocess.py
```

当前清洗规则：

- 删除空文本；
- 删除少于 `min_chars` 个字符的文本；
- 不要求文本必须包含中文，英文和中英混合文本都会保留；
- 默认保留繁体，不转换也不丢弃；
- 会把内部换行和连续空白压成单行，保证 `train.txt` 里一行是一条文档记录。

如需兼容手工 txt，可以把路径写进 `preprocess.raw_text_inputs`。这只是兼容入口，主路径仍然是从已下载 dataset 直接处理。

## 3. 生成 bin 文件

配置位置：`configs/data_pipeline.json` 的 `build_bin`。

运行：

```powershell
python .\data_pipeline\build_bin.py
```

该阶段会：

- 从 tokenizer 查询 EOS id；
- 根据词表大小选择 `uint16` 或 `uint32`；
- 保持输入顺序进行多进程编码；
- 先写临时文件，成功后再替换目标；
- 为每个 bin 写入 `.meta.json`，记录 token 数、dtype、EOS 和 tokenizer SHA256。

当前预训练加载代码固定按 `uint16` 读取，因此在词表超过 65,536、自动切换到 `uint32` 时，必须同步修改训练加载逻辑。
