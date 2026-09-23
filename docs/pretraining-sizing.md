# 预训练结构与训练预算审查

审查日期：2026-09-22。以 `models/` 和 `pretrain/run_train_model.py` 的实际实现为准；根目录 README 中“手写 AdamW、没有 nn.Linear”等介绍已经过时。本次只调整 `configs/pretrain.json`，不改词表、分词器或后训练配置。

## 已采用的结构

| 项目 | 调整前 | 调整后 |
| --- | ---: | ---: |
| 词表 | 24,576 | 24,576 |
| Transformer 层数 | 16 | 12 |
| 隐藏维度 | 1,024 | 576 |
| SwiGLU 中间维度 | 2,816 | 1,536 |
| 注意力头数 | 16 | 9 |
| 每头维度 | 64 | 64 |
| 训练序列长度 | 2,048 | 2,048 |
| 模型允许的最大长度 | 4,096 | 4,096 |
| 输入 embedding 与 lm_head | 共享 | 共享 |
| 唯一可训练参数 | 230,720,512 | 61,945,920 |

保留 decoder-only、pre-RMSNorm、RoPE（theta=10,000）、SwiGLU、无 bias 的线性层和全多头因果注意力 MHA。当前没有 GQA、MoE 或 dropout。每层两个残差分支，输出投影初始化按 `1/sqrt(2L)` 缩放。这里通过层数和宽度控制计算，避免引入新的网络实现。

576/1536/64 是可行的小模型维度组合，也见于 [SmolLM2-135M 的官方配置](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json)。本项目的层数、词表、MHA 和训练数据不同，不能据此推断能力等同。

参数公式（L=层数，d=隐藏维度，f=FFN 维度，V=词表）：

```text
P = V*d + L*(4*d*d + 3*d*f + 2*d) + d
```

| 参数组成 | 调整前 | 调整后 |
| --- | ---: | ---: |
| 共享 embedding / lm_head | 25,165,824 | 14,155,776 |
| 注意力 QKV 与输出投影 | 67,108,864 | 15,925,248 |
| SwiGLU 三个投影 | 138,412,032 | 31,850,496 |
| RMSNorm | 33,792 | 14,400 |

参数降至原来的 **26.85%**。共享参数只计一次；RoPE 缓存不属于可训练参数。

## 计算量是否达到目标

比较条件为相同训练 token 总数、相同 2,048 序列长度、相同精度，均不启用 activation checkpointing。不能仅把参数比例当成训练计算比例，也不能因为共享 embedding 就忽略输出词表投影。

主要矩阵乘法的前向加反向近似为：

```text
每 token 线性层训练 FLOPs ≈ 6 * [L*(4*d*d + 3*d*f) + V*d]
每 token 注意力矩阵训练 FLOPs ≈ 6*L*T*d（利用因果三角）
                            或 12*L*T*d（完整方阵计数）
```

| 每 token 的主要训练 FLOPs | 调整前 | 调整后 | 比例 |
| --- | ---: | ---: | ---: |
| 线性层 | 1.384 GFLOPs | 0.372 GFLOPs | 26.85% |
| 加上因果三角注意力 | 1.585 GFLOPs | 0.457 GFLOPs | 28.79% |
| 加上完整方阵注意力 | 1.787 GFLOPs | 0.541 GFLOPs | 30.30% |

两种计数方式都落在原计算量的 1/4～1/3 内。它们是矩阵乘法估算，不是对内核实耗的上下界；不含 RMSNorm、RoPE、激活、softmax/交叉熵、Muon 正交化、数据传输和保存等开销，也没有完整建模 FlashAttention 的重计算。因此不能承诺墙钟时间严格缩短到 29%～30%。实际时间要看训练日志的 `tokens_per_second`。

没有采用 12 层 × 640 × 1728：虽然参数降到了 32.60%，但计入 2K 注意力后的主要计算仍约为 34.41%～35.82%，超过 1/3 目标。

## 优化器与学习率

> 本节记录 MiniMind 接入前的历史配置。当前完整预训练版 `pretrain_t2t.jsonl` 的默认值为全 AdamW、warmup=500、micro-batch=8、每 500 步验证并保存。此前 mini 版一遍为 1,987 次更新，不代表完整版的训练步数。以下旧 Muon、1,000 步 warmup 和 5,000 步保存间隔不再是当前启动参数，详见 [预训练 README](../pretrain/README.md)。

当前实际使用的是 **torch.optim.Muon + torch.optim.AdamW**，并非所有参数都用 Muon，也不是根 README 所描述的手写 AdamW。

| 参数组 | 调整后参数数目 | 优化器 | 峰值 → 最小学习率 | weight decay |
| --- | ---: | --- | --- | ---: |
| block 内二维矩阵 | 47,775,744 | Muon | 0.02 → 0.002 | 0.0 |
| 共享 embedding / lm_head | 14,155,776 | AdamW | 0.0003 → 0.00003 | 0.1 |
| RMSNorm | 14,400 | AdamW | 0.0003 → 0.00003 | 0.0 |

Muon：momentum=0.95，Nesterov=true，Newton–Schulz 5 次，`adjust_lr_fn="original"`。AdamW：betas=(0.9, 0.95)，eps=1e-8，CUDA 上使用 fused 实现。累积完梯度后统一裁剪，max_norm=1.0。

参数分组合理：共享 embedding 没有被重复优化，归一化参数没有 weight decay。Muon 只用于隐藏层二维矩阵的做法与 [PyTorch 2.11 文档](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Muon.html) 一致；`original` 会按矩阵形状调整更新，不能直接把其 0.02 与 AdamW 的 0.0003 比大小。

本次保留现有优化器和学习率，作为保守起点。没有依据在尚未观察新结构 loss 曲线时提高学习率，也没有宣称这些值经过超参搜索达到最优。特别注意，顶层 `optimizer.weight_decay=0.1` 只影响 AdamW 的衰减组，Muon 使用自己配置中的 0.0。

两套学习率共用 1,000 次 optimizer update 的线性 warmup，之后分别余弦衰减到峰值的 10%。第 1 次更新的 AdamW/Muon LR 分别为 3e-7 / 2e-5；第 1,000 次更新达到峰值，最后一次更新达到最小值。调度单位是 optimizer update，不是 micro-batch 或 epoch。

## 训练轮次与数据预算

当前入口没有 `epochs`、`max_steps` 或 `max_train_tokens` 配置项。它按下面的公式自动训练约 **1 遍 token 文件**，然后结束：

```text
tokens_per_micro_batch = 4 * 2048 = 8,192
gradient_accumulation_steps = 131,072 / 8,192 = 16
total_updates = ceil(train_dataset_tokens / 131,072)
planned_tokens = total_updates * 131,072
```

loader 按连续窗口读取；文件尾部不足完整 micro-batch 的部分会被跳过并从头接续。因此这里的“一遍”是 token 数量近似，不保证每个尾部 token 恰好看一次。

历史日志 `output/train_logs/run_20260823_173959/config.json` 记录训练数据 13,567,229,433 tokens（约 135.67 亿），对应 103,510 次更新，计划 13,567,262,720 tokens，约 1.00000245 遍。对新模型约为 219 tokens/参数；warmup 约占更新数 0.97%，覆盖 131,072,000 tokens。**这是旧数据的示例，不是本次新语料的统计**。

推荐先保持现有的一遍训练，观察固定验证集 loss 和中文生成样例后决定是否继续；不自动增加重复轮数。重新构建 bin 后，入口会按实际数据量重新计算训练步数和余弦终点。若训练集不超过约 1.31 亿 tokens，固定 1,000 步 warmup 会触发入口校验，需要相应缩短 warmup；数据本身也不足以据此保证日常聊天质量。

本次将 eval_interval 从 1,000 改为 500，单次 train/val 各评估最多 65,536 tokens，使用固定分层窗口。checkpoint multiplier 仍为 10，因此每 5,000 次更新保存一次，最后一步也评估、保存。旧 2.307 亿参数 checkpoint 与新结构不兼容，保持 `paths.resume=null` 从头训练。

## 环境与启动

本机为 RTX 5060 Ti 8GB，Python 3.12 / PyTorch 2.11.0+cu128，支持 CUDA、BF16 和 Muon。但此 Windows PyTorch 构建没有编译 FlashAttention；原来的 `require_flash_attention=true` 在启动探测时必然失败。

现在设为 `false`，由 PyTorch SDPA 自动选择可用内核。这不是禁用 FlashAttention：若目标机器支持，SDPA 仍可选用它。BF16、batch=4、16 次梯度累积、131,072 tokens/update 和关闭 activation checkpointing 保持不变。8GB 机器若显存不足，可把 batch 改为 2，累积次数会自动变为 32，总 batch 和学习率无需随之改变。

审查时 `data_pipeline/data/train.bin`、`val.bin` 以及对应元数据在本机缺失。完整训练前需在训练机准备这四个文件，并保证元数据中的词表大小与 tokenizer SHA256 匹配 `bpe/tokenizer_24576`。现有分词器实测词表为 24,576，未修改。

在数据流水线已完成下载和预处理后，可用 `python data_pipeline/build_bin.py` 生成 bin。不要因为调整模型宽度重新训练 tokenizer；相同 tokenizer 的现有 bin 可以复用。

从仓库根目录启动：

```powershell
.\.venv\Scripts\python.exe -m pretrain.run_train_model --config configs/pretrain.json
```

其他已装好依赖的训练机使用：

```bash
python -m pretrain.run_train_model --config configs/pretrain.json
```

## 已完成的验证与能力边界

- 实例化实际模型并核对 61,945,920 个唯一参数、共享 embedding 身份和优化器分组。
- 使用本机 CUDA / BF16 / 自动 SDPA，按最终配置运行一次完整合成数据更新：batch=4、sequence=2048、累积 16 次、131,072 tokens；覆盖实际 `TokenBatchLoader`、前向、反向、梯度裁剪和 Muon/AdamW 更新。loss、梯度及更新后的参数均有限值，权重实际发生改变。
- 上述完整更新的 PyTorch 峰值 allocated 约 5.41 GiB、reserved 约 5.97 GiB，不包括桌面和其他进程显存。单次合成短测不能代表持续训练速度或长期稳定性。
- 验证学习率首步、warmup 终点和最终步数值；运行 `python -m unittest discover -s pretrain/tests -v`，已有 4 项验证测试全部通过。
- 未运行真实语料的 loss 收敛测试，也没有开始真实预训练；本机缺少配置指向的 bin 文件及其元数据。

约 6,195 万参数可作为低成本中文语言基础模型的实验起点；是否达到日常聊天要求，需要实际生成评测。预训练目标是 next-token prediction，不能保证只跑完预训练就具备稳定的多轮对话和指令遵循能力；例如 [SmolLM2 官方说明](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) 也将基础模型与经过对话 SFT 的 instruct 模型分开。本次不调整后训练。
