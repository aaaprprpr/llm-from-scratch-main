# 预训练

当前 `configs/pretrain.json` 的结构为 **12 层 / 576 隐藏维度 / 1536 FFN / 9 头 / 8,192 词表 / 32,768 上下文**，共 **52,508,736** 个参数。使用已有的 `tokenize/tokenizer`，不是重新运行 `tokenize/config.json` 中的 24K 分词器训练配方。当前 MiniMind 训练使用全 AdamW，已按 8K bin 训练约一遍，最终 checkpoint 为 `output/train_logs/run_20260924_005318/ckpt_step_14137.pt`。

旧版 24K 词表、2K 训练窗口的结构与计算量见 [历史预训练审查](../../docs/pretraining-sizing.md)；那份计算量比例不适用于当前 32K 训练。当前训练超参以本页和 `configs/pretrain.json` 为准；32K 训练要求 FlashAttention SDPA，启动时会校验支持情况。

## 当前 MiniMind 参数（RTX 5070 Ti 16GB 单卡）

| 项目 | 当前值 |
| --- | --- |
| 优化器 | AdamW；配置中保留的 `optimizer.muon` 字段不参与本次训练 |
| 学习率 | 500 次更新线性 warmup 到 3e-4，然后余弦降至 3e-5 |
| AdamW 参数 | betas=(0.9, 0.95)，eps=1e-8，矩阵 weight decay=0.1，norm 不衰减 |
| 梯度裁剪 | 1.0 |
| 模型上下文 / 训练序列长度 / micro-batch | 32,768 / 32,768 / 1 |
| 梯度累积 / 每次更新 token 数 | 4 / 131,072 |
| 精度与显存 | BF16、激活检查点、FlashAttention SDPA；CUDA 上使用 fused AdamW |
| 训练长度 | 约 1 遍完整 `pretrain_t2t.jsonl`；实际完成 14,137 次更新 |
| 验证与保存 | 每 500 次更新及最终一步验证、生成样例并保存 checkpoint |
| 恢复 | `paths.resume=null`，从头开始 |

此前 24K 词表、2K 窗口的 full 模型已完成 13,248 步预训练；[该版权重评测](../../docs/minimind-full-pretraining-evaluation-20260923.md)保留为历史记录。[两版最终权重对比](../../docs/minimind-24k-vs-8k-evaluation-20260927.md)记录了同文验证与长上下文探针。8K/32K 版是另一组从头训练的权重，不能普通 resume 旧 checkpoint。MiniMind 原文以短记录为主，连续 bin 的 32K 窗口会包含许多相邻但不相关的记录；跑完这份预训练不等于已经验证了 32K 长对话能力。

算力机为 RTX 5070 Ti 16GB、PyTorch 2.13.0+cu132，支持 BF16。原文件及两版 bin 和最终 checkpoint 已在本地。以后用同配置新开训练 run 的命令：

```bash
.venv/bin/python -m train.pretrain.run_train_model --config configs/pretrain.json
```

当前 batch_size 已为 1；如果 16GB 显存不足，需要进一步优化长序列损失计算或缩短训练窗口，不能再靠降低 batch_size 解决。保持 `paths.resume=null`。

```powershell
.\.venv\Scripts\python.exe -m train.pretrain.run_train_model --config configs/pretrain.json
```

当前使用 MiniMind 的主线完整预训练文件 `pretrain_t2t.jsonl`。数据源在 `configs/data_pipeline.json` 的 `sources.minimind` 定义；下载运行 `python -m dataset.data_pipeline.download`，bin 构建运行 `python -m dataset.data_pipeline.build_bin`。当前 8K bin 已生成，旧 24K bin 保留在原目录。具体用法见 [数据流水线](../../dataset/data_pipeline/README.md)。

## 预训练验证 loss

`configs/pretrain.json` 中的 `train.eval_tokens` 控制每次评估预算，`train.eval_seed` 固定评估抽样（缺省沿用训练 seed）。当前 65,536 tokens、序列长度 32,768，对应 2 个窗口。

训练启动时分别把 train/val token 文件的完整窗口划分为等大小区间，每个区间抽一个窗口；之后每次评估复用这些位置。这样覆盖整个文件，避免只反复评估开头连续的一段。窗口的目标 token 不重叠，较小文件不会为凑预算重复取样。使用独立 NumPy RNG，不推进训练游标，也不改变训练用随机状态。loss 按实际目标 token 总数加权，包括最后一个不满 batch 的评估批次。

每次运行日志目录中 `config.json` 的 `runtime.evaluation_sampling` 记录方法、种子、train/val 窗口位置和实际 token 数。固定文件、序列长度、预算与种子才能复现同一评估集合。重新生成 bin 或改变这些配置后，应将旧 checkpoint 在新集合上重新评估，不能直接比较旧前缀 loss 与新 loss 曲线。

此修复改善的是同一个验证文件中的抽样覆盖；它不会清洗验证集、消除训练集与验证集重复，也不能让 Wiki 验证集代表所有中文文本。来源分层、文档级去重分组切分、固定人工审核过的验证集仍需在构建数据时完成。现有 bin 没有文档/来源索引，暂不声称已实现来源级评估。

验证采样、目标偏移、加权和异常恢复：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s train\pretrain\tests -v
```
