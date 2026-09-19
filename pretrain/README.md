# 预训练验证 loss

`configs/pretrain.json` 中的 `train.eval_tokens` 控制每次评估预算，`train.eval_seed` 固定评估抽样（缺省沿用训练 seed）。当前 65,536 tokens、序列长度 2,048，对应 32 个窗口。

训练启动时分别把 train/val token 文件的完整窗口划分为等大小区间，每个区间抽一个窗口；之后每次评估复用这些位置。这样覆盖整个文件，避免只反复评估开头连续的一段。窗口的目标 token 不重叠，较小文件不会为凑预算重复取样。使用独立 NumPy RNG，不推进训练游标，也不改变训练用随机状态。loss 按实际目标 token 总数加权，包括最后一个不满 batch 的评估批次。

每次运行日志目录中 `config.json` 的 `runtime.evaluation_sampling` 记录方法、种子、train/val 窗口位置和实际 token 数。固定文件、序列长度、预算与种子才能复现同一评估集合。重新生成 bin 或改变这些配置后，应将旧 checkpoint 在新集合上重新评估，不能直接比较旧前缀 loss 与新 loss 曲线。

此修复改善的是同一个验证文件中的抽样覆盖；它不会清洗验证集、消除训练集与验证集重复，也不能让 Wiki 验证集代表所有中文文本。来源分层、文档级去重分组切分、固定人工审核过的验证集仍需在构建数据时完成。现有 bin 没有文档/来源索引，暂不声称已实现来源级评估。

验证采样、目标偏移、加权和异常恢复：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s pretrain\tests -v
```
