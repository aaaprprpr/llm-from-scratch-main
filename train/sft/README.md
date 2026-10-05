# SFT 使用说明

从仓库根目录运行以下命令，使用已安装项目依赖的 Python。配置入口是 [`configs/sft.json`](../../configs/sft.json)。

当前默认从 `run_20260923_130232/ckpt_step_13248.pt` 的 24K 预训练权重开始，使用 `tokenizer/bpe_24576` 和本目录的 `chat_template.jinja`。**模型结构由 checkpoint 的 `model_args` 决定**，不用在 SFT 配置中复制层数、宽度或词表大小。

## 数据准备与训练

默认数据是 COIG-CQIA，下载的子集由 `data.subsets` 指定，保存到 `paths.dataset`。自备数据应为 `datasets.save_to_disk` 格式，包含 `instruction`、`input`、`output` 字段。

```bash
python -m train.sft.download
python -m train.sft.sft
```

训练入口会构建 `paths.tokenized_cache`、划分验证集，只对 assistant 回答和结束标记计算损失。超长回答会跳过，不截断答案后继续训练。默认训练长度 2,048、micro-batch 1、梯度累积 16、2 个 epoch；完整 checkpoint、指标和生成样例写入 `paths.sft_logs`。

## 恢复与试玩

把 `paths.resume` 设为已有 SFT `ckpt_step_*.pt`，再次运行训练命令即可恢复模型、优化器、更新步数和数据位置。恢复不依赖原始预训练权重，但要保持训练数据、分词器、模板和训练设置一致；完成原定 epoch 的 checkpoint 不会自动多训练一轮。新训练时把 `paths.resume` 改回 `null`。

```bash
python -m train.sft.play_sft_model
```

试玩默认读取 `paths.sft_logs` 中最新 run 的最高步数 checkpoint，先运行 `play.prompts`；在交互终端中可继续输入问题。应使用训练该权重时的分词器与对话模板。

## 导出不含优化器的权重

```bash
python -m train.sft.export
```

输出到 `output/sft_weights/model.pt`，保留 `model`、`model_args`、配置和分词器指纹，旁边附带 `tokenizer/` 与 `chat_template.jinja`。这是项目格式的精简 checkpoint，不是只有张量的裸 state_dict；继续训练应使用训练日志中的完整 checkpoint。

截至 2026-10-05，本机缺少默认预训练 checkpoint 和 SFT 数据。本次只做了兼容修改与轻量合成检查，未启动真实 SFT 训练。
