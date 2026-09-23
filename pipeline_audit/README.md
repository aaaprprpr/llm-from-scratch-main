# Pretraining pipeline audit

这个目录只审计从 token binary 开始，到训练 batch、next-token 标签、模型前向和
checkpoint 的链路。预处理前后的正文质量不属于这里，标注清洗实验记录放在
[`label/experiments`](../label/experiments/README.md)。

- `run_audit.py`：依次执行四项审计。
- `audits/`：batching、model、bin、checkpoint 的具体检查。
- `common.py`：路径、报告和通用检查结果结构。
- `config.json`：本机输入路径和采样规模。
- `reports/run_<时间>/`：历史运行报告；这些报告由工具生成并被 Git 忽略。

审计代码不会修改训练数据、checkpoint 或生产训练代码。所有路径和采样规模集中在
`config.json`，不使用命令行参数。

运行：

```powershell
python pipeline_audit/run_audit.py
```

输出位于 `pipeline_audit/reports/run_<时间>/`：

- `summary.json`：所有 PASS/WARN/FAIL/SKIP 和数值指标。
- `bin_samples.html`：从不可直接阅读的 bin 中恢复出的文档和真实训练 batch。
- `checkpoint_metrics.csv`：真实 checkpoint 的位置 loss、错误标签基线和上下文消融。

本机没有复制 `train.bin` 时，bin 数据探针会显示 `SKIP`；没有配置 checkpoint 时，
checkpoint 探针也会显示 `SKIP`。总结果会显示 `INCOMPLETE`，合成 batch、标签、
因果遮罩和 KV-cache 检查仍会运行。

当前 `config.json` 的 bin 路径指向 MiniMind 全量预训练数据。`checkpoint` 默认为
`null`，因为本机没有与之对应的全量训练 checkpoint；要检查某个 checkpoint，先将
对应的 bin 和 checkpoint 放在本机，并填写 `paths.checkpoint`。不要用旧 FineWeb
checkpoint 对 MiniMind bin 做学习效果评估。
