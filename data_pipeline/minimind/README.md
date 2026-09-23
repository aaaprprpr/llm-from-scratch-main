# MiniMind 预训练数据入口

本目录包含 MiniMind 原始 JSONL 的校验、下载和 bin 转换代码。只生成预训练 bin，
不调用旧数据下载或正文清洗流程。现有数据路径保持兼容：

```text
data_pipeline/data/
  downloads/minimind/           # full/mini 原始 JSONL
  minimind_full/               # 当前 full 版 train.bin、val.bin 和元数据
  minimind/                    # 历史 mini 版 bin，不覆盖
```

路径由 `configs/data_pipeline.json` 的 `minimind_bin` 配置决定；当前预训练读
`minimind_full/`。整理代码时不迁移已有 bin，避免改变 checkpoint 所用数据路径。

## 当前启用：MiniMind 独立预训练入口

已按 `D:\minimind\README.md` 的数据用法接入 [MiniMind 数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)。当前默认使用主线完整预训练文件 `pretrain_t2t.jsonl`（8,275,074,893 字节），读取每行的 `{"text": "..."}`。同仓库的 `pretrain_t2t_mini.jsonl`（1,241,043,656 字节）是快速复现版，单独保存；上一轮 1,987 步训练只用过 mini。SFT、DPO、RL 文件不属于本次预训练输入。

上游 README 把 full 和 mini 作为主线版与轻量版的**替代选择**，并没有要求将两份文件拼接训练。两份原文件加起来约 9.52 GB；当前默认完整训练只读取 8.28 GB 的 full，避免可能重复采样 mini。原始文件的字节数和转成 `uint16` token 后的 bin 大小不可直接比较。

从仓库根目录运行正式入口：

```powershell
python -m data_pipeline.minimind.build_bin
```

旧命令 `python .\data_pipeline\build_minimind_bin.py` 保持兼容，调用同一实现。

配置在 `configs/data_pipeline.json` 的 **`minimind_bin`**。旧 `downloads` 和 `preprocess.dataset_sources` 中的所有数据源均已关闭；MiniMind 的下载由这个新入口负责，不走旧下载/清洗入口。

- 首次只下载配置选中的一个 JSONL；已有本地文件时直接复用，并验证配置中的 SHA256。full 和 mini 都可同时保存在原始数据目录中，但一次只选一个出 bin。文件和下载缓存都留在项目数据目录内。
- 数据集版本固定为 `312afb4f76391145c6902f765bb51691c09a12f5`，防止上游更新导致训练输入悄悄变化。
- 原样使用 `text`：不清洗、不去重、不修正文、不拼聊天模板，也不截断或补齐。格式损坏、缺失 `text` 或空字符串会报出行号并停止，不会静默丢弃记录。
- 保留现有 `bpe/tokenizer_24576`，不采用 MiniMind 自己的 tokenizer。用 seed=42 在记录层划分 95% 训练、5% 验证，再打乱并编码，每条记录末尾追加一个当前 tokenizer 的 EOS。
- 只在内存中保存行偏移，不生成清洗后的数据副本或 Arrow 副本。复用原 `build_bin.py` 的二进制写入函数；原下载、清洗、整理及通用出 bin 代码均未改动。
- 完整版产物为 `data_pipeline/data/minimind_full/train.bin`、`val.bin` 及各自的 `.meta.json`。旧 mini bin 仍保留在 `data_pipeline/data/minimind/`，不会覆盖。元数据包含记录/token 数、原文件 SHA256（`dataset_fingerprint`）、词表大小及 tokenizer SHA256。预训练配置已指向完整版 bin。
- 算力机实测：full 文件有 8,468,827 条记录；train 为 8,045,386 条、1,736,350,851 tokens（3,472,701,702 字节），val 为 423,441 条、91,375,436 tokens（182,750,872 字节）。两份原文件 SHA256 和 full bin 元数据均已核对。当前本机 C 盘没有额外保存 8.28 GB 原文件；完整原文件和 bin 在算力机的 `新加卷` 项目目录下。
- 默认 `overwrite=false`，已有产物时提前报错。确需重做时只修改 `minimind_bin.overwrite`。如已手动下载，把文件放到 `data_pipeline/data/downloads/minimind/`；可设 `download=false`，完全离线转换。

上游建议 mini 的 `max_seq_len≈768`、full 的 `max_seq_len≈380`，对应 MiniMind 自己的 tokenizer 及逐条截断/补齐的训练方式。本项目仍采用连续 token bin，由现有训练代码取窗口，因此转换阶段不照搬这些截断值。这里没有混入 SFT 数据。

如需重做轻量实验，可将 `filename` 改为 `pretrain_t2t_mini.jsonl`、`sha256` 改为 `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`，把 train/val 路径指向新目录，并同步修改 `configs/pretrain.json` 的数据路径。不要覆盖已经训练过的 bin 和 checkpoint。

