# 数据流水线

数据相关代码统一分为三个阶段。所有命令都从仓库根目录执行。

## 1. 下载数据集

保存 Hugging Face 原始数据集，适合 SFT 等保留结构化字段的场景：

```powershell
python -m data_pipeline.download `
  --dataset llamafactory/alpaca_zh `
  --output data/downloads/alpaca_zh
```

将数据集字段导出成预训练用 UTF-8 文本：

```powershell
python -m data_pipeline.download `
  --dataset <dataset-id> `
  --split train `
  --format text `
  --text-columns text `
  --output data/raw/corpus.txt
```

建议固定 `--revision`，避免远端数据更新后无法复现实验。

## 2. 数据预处理

默认行为：

- 流式读取 `data/raw/*.txt`，不会将整个语料载入内存；
- 使用 ftfy 修复文本（未安装时给出警告并跳过）；
- 删除空行、少于 6 个字符的行和不含中文的行；
- 默认保留繁体；可显式选择丢弃或转换；
- 使用固定随机种子按行划分 train/val；
- 输出 `data/train.txt`、`data/val.txt` 和 `data/preprocess.meta.json`。

```powershell
python -m data_pipeline.preprocess `
  --input "data/raw/*.txt" `
  --train-output data/train.txt `
  --val-output data/val.txt `
  --train-ratio 0.95 `
  --seed 42 `
  --traditional-mode convert
```

如需复刻旧脚本“发现繁体就删除整行”的行为，使用：

```powershell
python -m data_pipeline.preprocess --traditional-mode drop
```

如需同时准备 tokenizer 训练文本，可添加：

```powershell
--bpe-chunk-dir bpe/data --bpe-chunk-size-mb 100
```

BPE chunk 只取 train 部分，不再使用 val 语料训练 tokenizer。

## 3. 生成 bin 文件

```powershell
python -m data_pipeline.build_bin `
  --tokenizer bpe/tokenizer `
  --train-text data/train.txt `
  --val-text data/val.txt `
  --train-bin data/train.bin `
  --val-bin data/val.bin `
  --workers 6
```

该阶段会：

- 从 tokenizer 查询 EOS id，不再硬编码为 0；
- 根据词表大小选择 `uint16` 或 `uint32`；
- 保持输入顺序进行多进程编码；
- 先写临时文件，成功后再替换目标；
- 为每个 bin 写入 `.meta.json`，记录 token 数、dtype、EOS 和 tokenizer SHA256。

当前预训练加载代码固定按 `uint16` 读取，因此在词表超过 65,536、自动切换到 `uint32` 时，必须同步修改训练加载逻辑。

## 目录约定

```text
data_pipeline/
├── download.py       # 下载或导出数据集
├── preprocess.py     # 清洗、繁体策略、划分、BPE 文本分块
└── build_bin.py      # tokenizer 编码和 bin 生成

data/
├── raw/              # 下载/手工准备的 UTF-8 原始文本
├── train.txt         # 预处理产物（通常被 gitignore）
├── val.txt           # 预处理产物（通常被 gitignore）
├── train.bin         # 训练 token 流
└── val.bin           # 验证 token 流
```

