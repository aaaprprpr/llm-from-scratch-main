# Pretraining pipeline audit

这个目录只审计从 token binary 开始，到训练 batch、next-token 标签、模型前向和
checkpoint 的链路。预处理前后的正文质量不属于这里；后续应由单独的数据浏览和人工
标注工具负责。

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

本机没有复制 `train.bin` 时，bin 和 checkpoint 数据探针会明确显示 `SKIP`，不把缺少
外部大文件误报为代码失败；总结果会显示 `INCOMPLETE`，合成 batch、标签、因果遮罩和
KV-cache 测试仍会运行。
