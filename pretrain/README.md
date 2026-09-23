# 预训练

当前默认配置为 **12 层 / 576 隐藏维度 / 1536 FFN / 9 头 / 24,576 词表**，共 **61,945,920** 个参数。相对原 16 层 / 1024 / 2816 配置，在相同 2K 长度和 token 预算下，主要训练矩阵计算约为 **28.8%～30.3%**。当前 MiniMind 训练使用全 AdamW，按实际数据量训练约一遍。

结构与计算量推导见 [历史预训练审查](../docs/pretraining-sizing.md)；当前训练超参以本页和 `configs/pretrain.json` 为准。默认允许 SDPA 自动选择注意力内核。

## 当前 MiniMind 参数（RTX 5070 Ti 16GB 单卡）

| 项目 | 当前值 |
| --- | --- |
| 优化器 | AdamW；配置中保留的 `optimizer.muon` 字段不参与本次训练 |
| 学习率 | 500 次更新线性 warmup 到 3e-4，然后余弦降至 3e-5 |
| AdamW 参数 | betas=(0.9, 0.95)，eps=1e-8，矩阵 weight decay=0.1，norm 不衰减 |
| 梯度裁剪 | 1.0 |
| 序列长度 / micro-batch | 2048 / 8 |
| 梯度累积 / 每次更新 token 数 | 8 / 131,072 |
| 精度 | BF16；CUDA 上使用 fused AdamW |
| 训练长度 | 约 1 遍完整 `pretrain_t2t.jsonl`；1,736,350,851 个训练 tokens 对应 13,248 次更新 |
| 验证与保存 | 每 500 次更新及最终一步验证、生成样例并保存 checkpoint |
| 恢复 | `paths.resume=null`，从头开始 |

上一轮只训练了轻量版 mini（260,350,331 个训练 tokens、1,987 步），不是完整预训练文件。当前已将数据入口切到主线 full，保留旧 bin 和 checkpoint；full 和 mini 在上游 README 中是替代选择，不混合训练。旧 run 的 Muon 配置出现过明显退化，本次仍以 AdamW 为起点；这些参数未经 full 版收敛实验确定最优。每份 checkpoint 约 0.74 GB，预留足够磁盘空间。

算力机已确认是 RTX 5070 Ti 16GB、PyTorch 2.13.0+cu132，支持 BF16。完整版原文件及独立 full bin 已在算力机生成并校验。算力机跟踪配置 `configs/pretrain.json` 目前仍指向 mini；在新的本地配置提交推送并拉取前，算力机应显式指定已保存的 full 配置：

```bash
.venv/bin/python -m pretrain.run_train_model --config output/minimind_full_pretrain.json
```

拉取本地更新后的配置后，才可以使用默认配置路径：

```bash
.venv/bin/python -m pretrain.run_train_model --config configs/pretrain.json
```

更小显存可将 `batch_size` 改为 4，累积自动变为 16，总 token batch 和学习率不变。不要用旧 Muon checkpoint 普通 resume 到本次 AdamW 训练。

```powershell
.\.venv\Scripts\python.exe -m pretrain.run_train_model --config configs/pretrain.json
```

当前只启用 MiniMind 的主线完整预训练文件 `pretrain_t2t.jsonl`。先运行 `python -m data_pipeline.minimind.build_bin`，生成 `data_pipeline/data/minimind_full/train.bin`、`val.bin` 和对应 `.meta.json`；旧命令 `python data_pipeline/build_minimind_bin.py` 仍可用。具体用法见 [MiniMind 数据入口](../data_pipeline/minimind/README.md)。词表仍为 24,576，原清洗流程不参与。保持 `paths.resume=null`；不要把 mini 版末尾 checkpoint 直接作为新一遍 full 版训练的普通 resume。

## 预训练验证 loss

`configs/pretrain.json` 中的 `train.eval_tokens` 控制每次评估预算，`train.eval_seed` 固定评估抽样（缺省沿用训练 seed）。当前 65,536 tokens、序列长度 2,048，对应 32 个窗口。

训练启动时分别把 train/val token 文件的完整窗口划分为等大小区间，每个区间抽一个窗口；之后每次评估复用这些位置。这样覆盖整个文件，避免只反复评估开头连续的一段。窗口的目标 token 不重叠，较小文件不会为凑预算重复取样。使用独立 NumPy RNG，不推进训练游标，也不改变训练用随机状态。loss 按实际目标 token 总数加权，包括最后一个不满 batch 的评估批次。

每次运行日志目录中 `config.json` 的 `runtime.evaluation_sampling` 记录方法、种子、train/val 窗口位置和实际 token 数。固定文件、序列长度、预算与种子才能复现同一评估集合。重新生成 bin 或改变这些配置后，应将旧 checkpoint 在新集合上重新评估，不能直接比较旧前缀 loss 与新 loss 曲线。

此修复改善的是同一个验证文件中的抽样覆盖；它不会清洗验证集、消除训练集与验证集重复，也不能让 Wiki 验证集代表所有中文文本。来源分层、文档级去重分组切分、固定人工审核过的验证集仍需在构建数据时完成。现有 bin 没有文档/来源索引，暂不声称已实现来源级评估。

验证采样、目标偏移、加权和异常恢复：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s pretrain\tests -v
```
