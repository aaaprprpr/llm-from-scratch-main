# 数据流水线

数据相关代码统一从 `configs/data_pipeline.json` 读取配置，不再传命令行参数。所有命令都从仓库根目录执行。

| 目录或文件 | 职责 |
| --- | --- |
| `minimind/` | 当前 MiniMind 预训练 JSONL 校验、下载和 bin 转换 |
| `build_minimind_bin.py` | 旧命令的兼容入口 |
| `download.py`、`preprocess.py`、`build_bin.py` | 旧通用流程；数据源当前全部关闭 |
| `record_adapters.py` | 旧通用流程及标注工具共用的字段适配 |
| `dataset_samples/` | 旧下载流程生成的少量结构样本，不参与训练 |
| `data/` | 原始文件、bin、元数据和临时缓存 |

## 当前启用：MiniMind 独立预训练入口

MiniMind 的原始 JSONL 校验、下载和出 bin 实现在 [`minimind/`](minimind/README.md)。
旧命令 `python .\data_pipeline\build_minimind_bin.py` 继续可用；正式入口是
`python -m data_pipeline.minimind.build_bin`。当前配置只启用 MiniMind，
原始文件和 full/mini bin 沿用现有路径，不迁移已生成的训练数据。

## 原有三步流程（当前未启用）

下面保留原流程说明。旧目录中的数据不会混入上述 MiniMind 入口。

```text
download.py    -> 下载/管理原始 dataset，并生成结构样本
preprocess.py  -> adapter 格式化、清洗，输出完整 text 记录
build_bin.py   -> 记录级 train/val 划分、分词，生成连续 token bin
```

## 1. 下载数据集

配置位置：`configs/data_pipeline.json` 的 `downloads`。

`download.py` 只负责三件事：

- 按 `downloads` 里的启用项下载 Hugging Face dataset；
- 如果目标目录已经存在且 Dataset 可正常加载，标记为 `already_downloaded`；中断留下的不完整目录会重新下载/保存；
- 整轮下载成功后删除 Hugging Face 缓存，项目里只留下 `downloads/<source_id>`（见下面的缓存说明）；
- 在 `data_pipeline/dataset_samples/` 下生成 `<source_id>.sample.json`，方便查看字段结构和少量样本。

运行：

```powershell
python .\data_pipeline\download.py
```

下面是保留的旧数据源配置说明。FineWeb、FineWiki、Wikipedia、TigerResearch 等旧数据源当前全部禁用；只有手动重新启用后，旧下载入口才会处理它们。

| 配置中的 source_id | 下载范围 | 全部分片数 | Parquet 下载大小 |
| --- | --- | ---: | ---: |
| `fineweb_edu_zh_4_5` | `opencsg/Fineweb-Edu-Chinese-V2.1` 的 `4_5/*.parquet` | 9,745 | 74.29 GB（69.19 GiB） |
| `finewiki_zh` | `HuggingFaceFW/finewiki` 的 `data/zhwiki/*.parquet` | 5 | 5.53 GB（5.15 GiB） |

大小是 2026-09-19 对官方文件列表分页求和的结果，分别为 74,294,352,764 和 5,526,531,060 字节，总计约 79.82 GB。官方说明中的约 70 GB / 5.1 GB 与此处文件字节统计不同；本流程还会生成 Arrow 缓存和 `save_to_disk` 副本，最终磁盘占用高于下载大小。文件列表：[FineWeb 4_5](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1/tree/main/4_5)、[FineWiki 中文](https://huggingface.co/datasets/HuggingFaceFW/finewiki/tree/main/data/zhwiki)。

这两个数据源通过 `data_files` 限定完整目标子集；代码会拒绝缺失或扩大范围的配置。FineWiki 的加载配置名是 `zh`，文件目录是 `data/zhwiki`。它们都保存为现有的 `format: "disk"`，预处理配置保留 `text_only` adapter：只读取清洗后的 `text`，不拼接标题或读取 `wikitext`；当前下载和预处理开关均已关闭。

注意：

- Hugging Face 数据源统一使用 `format: "disk"`，下载为 `data_pipeline/data/downloads/<source_id>`。
- `download.py` 的 Hub 原文件、Datasets Arrow、下载/解压、Xet 等缓存统一固定在项目内的 `data_pipeline/data/.cache/huggingface/`；路径基于脚本位置解析，从其他工作目录启动也不会改变。脚本在导入 Hugging Face 库前覆盖本进程的缓存路径，并显式传入 `cache_dir`，不会把语料缓存写进用户目录下的默认 `.cache/huggingface/`。这不修改系统环境变量或登录凭证，也不搬移、删除已有的默认缓存。
- Hugging Face 的加载路径必然是「先落缓存，再 `save_to_disk`」，没有直接写进目标目录的方式。所以缓存只在下载期间存在：`download.py` 在**整轮下载全部成功**后递归删除 `data_pipeline/data/.cache/huggingface/`，跑完项目里只剩 `downloads/<source_id>`。中途抛异常时**不**清理，重跑可直接复用缓存、不必重新下载几十 GB；此时日志会打印缓存路径，可手动删除。把 `download_cache_cleanup` 设为 `false` 可关闭自动清理，缓存保留在项目内。清理只删缓存目录，不动 `downloads/` 下已保存的数据集。
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
