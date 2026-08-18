# 数据流水线

数据相关代码统一从 `configs/data_pipeline.json` 读取配置，不再传命令行参数。所有命令都从仓库根目录执行。

整体流程：

```text
download.py    -> 下载/管理原始 dataset，并生成结构样本
preprocess.py  -> adapter 格式化、清洗，输出完整 text 记录
build_bin.py   -> 记录级 train/val 划分、分词，生成连续 token bin
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

- Hugging Face 数据源统一使用 `format: "disk"`，下载为 `data_pipeline/data/downloads/<source_id>`。
- `download.py` 不再导出 txt。
- 下载前会打印 dataset、split、预估总量和保存目录；下载时 Hugging Face 会显示当前文件名及真实字节数。
- `wikimedia/wikipedia` 必须显式写 `config: "20231101.zh"` 和 `split: "train"`。不要把 wiki 配成 `config: null`，避免误拉全量 wikipedia。

## 2. 数据预处理

配置位置：`configs/data_pipeline.json` 的 `preprocess`。

`preprocess.py` 负责：

- 读取 `preprocess.dataset_sources` 中启用的数据源；
- 从对应的 `data_pipeline/data/downloads/...` 目录加载 dataset；
- 按每个数据源的 `adapter` 转成完整的预训练文本记录；
- 清洗文本；预训练数据量较大，因此保留重复记录，不做全局去重；
- 输出只有 `text` 列的 `data_pipeline/data/preprocessed` Arrow Dataset；
- 将过滤原因的汇总计数写入 `data_pipeline/data/preprocessed.report.json`，不保存被过滤正文。

运行：

```powershell
python .\data_pipeline\preprocess.py
```

这一阶段不做 train/val 划分、分词或训练长度切块，记录内部的段落和对话轮次不会被拆开。

## 3. 生成 bin 文件

配置位置：`configs/data_pipeline.json` 的 `build_bin`。

运行：

```powershell
python .\data_pipeline\build_bin.py
```

该阶段会：

- 直接读取 `data_pipeline/data/preprocessed`，不经过 txt；
- 用固定 seed 在完整记录层划分 train/val，按随机块顺序读取并在块内打乱；
- 从 tokenizer 查询 EOS id；
- 完整编码每条记录，只在记录末尾追加一次 EOS；
- 多进程编码后分别写入无文件头的连续 token 流；
- 先写临时文件，成功后再替换目标；
- 为每个 bin 写入 `.meta.json`，记录划分参数、记录数、token 数、dtype、EOS 和 tokenizer SHA256。

当前预训练加载代码按 `uint16` 读取，因此配置也固定为 `uint16`；词表超过 65,536 时需要同时修改训练端。
