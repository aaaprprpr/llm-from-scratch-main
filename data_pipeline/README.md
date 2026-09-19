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
- 如果目标目录已经存在且 Dataset 可正常加载，标记为 `already_downloaded`；中断留下的不完整目录会重新下载/保存，并复用 Hugging Face 缓存；
- 在 `data_pipeline/dataset_samples/` 下生成 `<source_id>.sample.json`，方便查看字段结构和少量样本。

运行：

```powershell
python .\data_pipeline\download.py
```

已启用 FineWeb 高分档和 FineWiki 中文版，默认下载下表中目标目录的**全部分片**，无需修改分片数量。原有 Wikipedia、TigerResearch 启用项仍保留，因此运行命令也会处理它们。

| 配置中的 source_id | 下载范围 | 全部分片数 | Parquet 下载大小 |
| --- | --- | ---: | ---: |
| `fineweb_edu_zh_4_5` | `opencsg/Fineweb-Edu-Chinese-V2.1` 的 `4_5/*.parquet` | 9,745 | 74.29 GB（69.19 GiB） |
| `finewiki_zh` | `HuggingFaceFW/finewiki` 的 `data/zhwiki/*.parquet` | 5 | 5.53 GB（5.15 GiB） |

大小是 2026-09-19 对官方文件列表分页求和的结果，分别为 74,294,352,764 和 5,526,531,060 字节，总计约 79.82 GB。官方说明中的约 70 GB / 5.1 GB 与此处文件字节统计不同；本流程还会生成 Arrow 缓存和 `save_to_disk` 副本，最终磁盘占用高于下载大小。文件列表：[FineWeb 4_5](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/tree/main/4_5)、[FineWiki 中文](https://huggingface.co/datasets/HuggingFaceFW/finewiki/tree/main/data/zhwiki)。

这两个数据源通过 `data_files` 限定完整目标子集；代码会拒绝缺失或扩大范围的配置。FineWiki 的加载配置名是 `zh`，文件目录是 `data/zhwiki`。它们都保存为现有的 `format: "disk"`，后续 `preprocess` 已启用并使用 `text_only` adapter：只读取清洗后的 `text`，不拼接标题或读取 `wikitext`。

注意：

- Hugging Face 数据源统一使用 `format: "disk"`，下载为 `data_pipeline/data/downloads/<source_id>`。
- `download.py` 的 Hub 原文件、Datasets Arrow、下载/解压、Xet 等缓存统一固定在项目内的 `data_pipeline/data/.cache/huggingface/`；路径基于脚本位置解析，从其他工作目录启动也不会改变。脚本在导入 Hugging Face 库前覆盖本进程的缓存路径，并显式传入 `cache_dir`，不会把语料缓存写进用户目录下的默认 `.cache/huggingface/`。这不修改系统环境变量或登录凭证，也不搬移、删除已有的默认缓存。
- `download.py` 不再导出 txt。
- 下载前会打印 dataset、split、预估总量、保存目录和缓存根目录；下载时 Hugging Face 会显示当前文件名及真实字节数。
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
