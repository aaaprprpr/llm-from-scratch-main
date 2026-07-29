看完了，没有改代码。总体判断是：`models/model.py` 不需要推倒重写。当前已经是比较正常的 dense decoder：12 层、512 维、8 头 MHA、Pre-RMSNorm、SwiGLU、RoPE、PyTorch SDPA，总计约 5873 万参数。主要缺口是长上下文相关的工程实现还比较初级。


二、KV cache 已经支持，而且数值逻辑是正确的。完整前向、逐 token cache、分块 cache 的误差都在 `1e-6` 内。不过它现在属于“能用但不高效”：每一步都把历史 K/V 与新 K/V 重新 `torch.cat`，累计产生 O(T²) 的复制；没有静态预分配、cache position、滑动窗口或者 paged cache。当前 MHA 在 BF16、batch=1、4K 上下文时，12 层 KV cache 大约 96 MiB。

要区分清楚：KV cache 只优化自回归生成，不能降低预训练的显存和计算量。预训练长上下文更依赖 FlashAttention、BF16 和 activation checkpointing。

三、当前已经是标准 8 头 MHA，不是单头。实际最值得升级的是 GQA，而不是立刻做 MLA。保留 8 个 query heads，把 KV heads 改成 2，KV cache 可以直接降到四分之一，模型改动也比较有限；GQA 本身就是为接近 MHA 质量、接近 MQA 推理效率设计的。[GQA 原论文](https://arxiv.org/abs/2305.13245)

MLA 对 KV cache 压缩更狠，但需要低秩 KV latent、拆分 RoPE 维度以及专门的推理实现，和现有 SDPA 的衔接复杂很多。对于当前 5900 万参数模型，先做 GQA 的性价比明显更高。[DeepSeek-V2 的 MLA 与 DeepSeekMoE](https://arxiv.org/abs/2405.04434)

四、现在已经在调用 `scaled_dot_product_attention`，所以模型接口本身兼容 FlashAttention，正常无 mask 的预训练路径也比较干净。但训练入口目前默认 FP32，没有 autocast/BF16，因此不能说实际已经命中了 Flash 内核。PyTorch 会根据 GPU、dtype 和张量条件自动选择 FlashAttention、memory-efficient 或普通实现。[PyTorch SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)

FlashAttention 主要解决 T×T 注意力矩阵的显存和 IO，不改变 O(T²) 计算量。256 扩到 2048，单样本注意力计算约增加 64 倍；扩到 4096 则约增加 256 倍。所以 BF16、梯度累积、activation checkpointing 和 token-based 训练预算也要同步调整，但这些应该放在训练代码里，不应塞进 `model.py`。

五、Kimi 的 Attention Residuals 和 DeepSeek 的 mHC 都是残差拓扑，不是序列注意力。当前模型就是普通的 `x + Attention`、`x + FFN`。AttnRes 改为沿网络深度，对 embedding 和先前子层输出做注意力聚合；当前 12 个 block、24 个子层不算深，Full AttnRes 已经具备实现和实验价值，Block AttnRes 可以后面再做。它比 mHC 更适合作为第一个研究分支。[AttnRes 论文](https://arxiv.org/abs/2603.15031)、[官方仓库](https://github.com/MoonshotAI/Attention-Residuals)

DeepSeek 那个准确名称是 mHC，即 Manifold-Constrained Hyper-Connections。它把 residual stream 扩展成多条并行流，再通过 Sinkhorn 投影约束动态混合矩阵。论文中的效率依赖融合内核、重计算和并行系统；直接用普通 PyTorch 实现，会显著增加残差激活和内存带宽，而长上下文本来就缺显存。因此可以实现，但不适合第一轮作为默认结构。[mHC 论文](https://arxiv.org/abs/2512.24880)

AttnRes 和 mHC 应当做成互斥实验项，分别与标准残差进行同预算对照，不建议一开始叠加，否则很难判断收益来自哪里。

六、MoE 当前不值得优先做。它解决的是总参数容量与激活计算量之间的关系，不解决上下文长度、注意力 O(T²) 或 KV cache。当前模型规模和单机训练条件下，router、负载均衡、专家分发以及额外优化器状态很可能让训练更慢。等 dense 长上下文基线稳定后，再把 SwiGLU 替换成小型 top-2 MoE 做研究更合理。

另外还有三个比 MoE 更值得先处理的结构细节：当前没有统一的权重初始化；SwiGLU 的 `d_ff=2048=4*d_model`，使 FFN 占每个 block 约 75% 参数，属于偏宽配置而不是常见的等参数 SwiGLU；embedding 和 lm_head 没有绑权，额外占约 419 万参数。这些不是必然错误，但重构时应该明确决定，而不是继续依赖默认值。

我的建议顺序是：先做 2K/4K dense 长上下文基线，整理共享 RoPE、GQA、静态 KV cache和明确初始化；训练侧接上 BF16、Flash 实际派发、梯度累积与 activation checkpointing；基线稳定后先实验 AttnRes，再单独实验 mHC；MLA、MoE、Kimi Linear 暂时放到后面的独立分支。这样每次结构变化都能测出真实收益，也不会把长上下文、推理缓存和研究型残差混成一团。
