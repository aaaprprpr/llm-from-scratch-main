# 预训练

当前 `configs/pretrain.json` 使用 **24,576 词表 / 4,096 模型上下文 / 2,048 训练序列**。模型为 12 层、576 隐藏维度、1536 FFN、9 头，共 **61,945,920** 个参数；分词器为 `tokenizer/bpe_24576`，训练和验证 bin 为 `dataset/data_pipeline/data/minimind_full/`。这些参数与 `output/train_logs/run_20260923_130232/ckpt_step_13248.pt` 记录的 24K 训练版一致。当前 `paths.resume=null`，启动命令会新建 run。`play_model.py` 和 `export.py` 的默认 checkpoint 已指向这版最终权重。

## 当前 MiniMind 参数（RTX 5070 Ti 16GB 单卡）

| 项目 | 当前值 |
| --- | --- |
| 优化器 | AdamW；配置中的 `optimizer.muon` 字段未启用 |
| 学习率 | 500 次更新线性 warmup 到 3e-4，然后余弦降至 3e-5 |
| AdamW 参数 | betas=(0.9, 0.95)，eps=1e-8，矩阵 weight decay=0.1，norm 不衰减 |
| 梯度裁剪 | 1.0 |
| 模型上下文 / 训练序列 / micro-batch | 4,096 / 2,048 / 8 |
| 梯度累积 / 每次更新 token 数 | 8 / 131,072 |
| 精度与显存 | BF16；无需激活检查点或强制 FlashAttention；CUDA 上使用 fused AdamW |
| 训练长度 | 约 1 遍完整 `pretrain_t2t.jsonl`；已有 24K run 完成 13,248 次更新 |
| 验证与保存 | 每 500 次更新及最终一步验证、生成样例并保存 checkpoint |
| 恢复 | `paths.resume=null`，从头开始 |

[24K 最终权重评测](../../docs/历史记录/2026-09-23-MiniMind完整版预训练评测.md)和[24K/8K 对比](../../docs/历史记录/2026-09-27-MiniMind两版词表与上下文对比.md)记录了既有结果。8K/32K 实验的 checkpoint 和 bin 保留在各自原目录。

从仓库根目录启动一轮新训练：

```bash
.venv/bin/python -m train.pretrain.run_train_model --config configs/pretrain.json
```

```powershell
.\.venv\Scripts\python.exe -m train.pretrain.run_train_model --config configs/pretrain.json
```

原始 MiniMind 完整预训练文件 `pretrain_t2t.jsonl` 定义在 `configs/data_pipeline.json` 的 `sources.minimind`。`build_bin` 现也指向 24K 分词器和 `minimind_full/`，已经有 bin 时 `overwrite=false` 会阻止覆盖。具体用法见[数据流水线](../../dataset/data_pipeline/README.md)。

## 预训练验证 loss

`configs/pretrain.json` 中的 `train.eval_tokens` 控制每次评估预算，`train.eval_seed` 固定评估抽样（缺省沿用训练 seed）。当前 65,536 tokens、序列长度 2,048，对应 32 个窗口。

训练启动时分别把 train/val token 文件的完整窗口划分为等大小区间，每个区间抽一个窗口；之后每次评估复用这些位置。这样覆盖整个文件，避免只反复评估开头连续的一段。窗口的目标 token 不重叠，较小文件不会为凑预算重复取样。使用独立 NumPy RNG，不推进训练游标，也不改变训练用随机状态。loss 按实际目标 token 总数加权，包括最后一个不满 batch 的评估批次。

每次运行日志目录中 `config.json` 的 `runtime.evaluation_sampling` 记录方法、种子、train/val 窗口位置和实际 token 数。固定文件、序列长度、预算与种子才能复现同一评估集合。重新生成 bin 或改变这些配置后，应将旧 checkpoint 在新集合上重新评估，不能直接比较旧前缀 loss 与新 loss 曲线。

此修复改善的是同一个验证文件中的抽样覆盖；它不会清洗验证集、消除训练集与验证集重复，也不能让 Wiki 验证集代表所有中文文本。来源分层、文档级去重分组切分、固定人工审核过的验证集仍需在构建数据时完成。现有 bin 没有文档/来源索引，暂不声称已实现来源级评估。

验证采样、目标偏移、加权和异常恢复：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s train\pretrain\tests -v
```
