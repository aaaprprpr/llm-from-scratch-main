# DPO 使用说明

从仓库根目录运行命令，配置入口是 [`configs/dpo.json`](../../configs/dpo.json)。DPO 使用已完成的 SFT 权重，**模型结构以 checkpoint 的 `model_args` 为准**，不用手抄 `model` 配置。当前分词器为 `tokenizer/bpe_24576`，模板沿用 SFT 的 `chat_template.jinja`；选择权重时，应使用它训练时的分词器与模板。

## 数据准备与训练

`paths.dataset` 默认是 `wenbopan/Chinese-dpo-pairs`：第一次训练时由 `datasets` 加载并缓存。也可以填本地 `datasets.save_to_disk` 数据目录；必须包含 `prompt`、`chosen`、`rejected`，可选 `system`，由 `data.split` 选择划分。

用 `paths.sft_checkpoint` 指定作为起点的完整 SFT checkpoint；为 `null` 时，从 `paths.sft_logs` 自动选择最新 run 的最高步数 checkpoint。训练时同时建立可更新的 policy 和冻结的 SFT reference。

```bash
python -m train.dpo.dpo
```

入口会构建 `paths.tokenized_cache`、划分验证集；chosen 和 rejected 都只统计 assistant 回答，超长答案会跳过。默认最大长度 2,048、micro-batch 2、梯度累积 4、1 个 epoch。完整 checkpoint、指标和生成样例写入 `paths.dpo_logs`。

## 恢复与试玩

把 `paths.resume` 设为已有 DPO `ckpt_step_*.pt`，再次运行训练命令。恢复时仍使用记录路径中的 SFT reference，请保留这份权重。数据、分词器、模板和训练设置应与原 run 保持一致；新训练把 `paths.resume` 改回 `null`。

```bash
python -m train.dpo.play_dpo_model
```

`paths.dpo_checkpoint` 可指定试玩和导出的权重；为 `null` 时选择 `paths.dpo_logs` 中最新 run 的最高步数 checkpoint。试玩先生成 `play.prompts`，交互终端中可继续输入问题。

## 导出不含优化器的权重

```bash
python -m train.dpo.export
```

输出由 `paths.clean_weights` 指定，默认 `output/dpo_weights/model.pt`。精简 checkpoint 保留 `model`、`model_args`、配置、stage 和分词器 SHA，同目录附带 `tokenizer/` 与 `chat_template.jinja`；继续训练仍使用完整 DPO checkpoint。这份项目格式权重还可由 `export/hf/export_dpo.py` 转成标准 Transformers 格式。

截至 2026-10-05，本机缺少当前主线的 base 权重、SFT checkpoint 和训练数据。本次完成的是链路兼容与轻量检查，未启动真实 SFT 或 DPO 训练。
