




四、数据搬运是同步的。每个微批次都在主线程完成 NumPy 转换，然后执行阻塞式 `.to(device)`，CPU 准备和 GPU 计算无法重叠。输入 token 很少，因此不需要上复杂的多进程 DataLoader；后续做一个简单的锁页双缓冲或预取就够了。



七、`torch.save` 直接同步保存 GPU 上的模型和 AdamW 状态，保存期间训练完全停止。这属于阶段性停顿，不影响单步计算；可以以后再考虑异步 CPU 快照，当前不值得先复杂化。

八、当前没有 `torch.compile`。手写 RoPE、RMSNorm、残差和 SwiGLU 都有不少零碎 kernel，compile 有实际价值；AdamW 也可以测试 fused 版本。但这两项属于完成前面清理后的第二阶段，不是现有 bug。










五、Kimi 的 Attention Residuals 和 DeepSeek 的 mHC 都是残差拓扑，不是序列注意力。当前模型就是普通的 `x + Attention`、`x + FFN`。AttnRes 改为沿网络深度，对 embedding 和先前子层输出做注意力聚合；当前 12 个 block、24 个子层不算深，Full AttnRes 已经具备实现和实验价值，Block AttnRes 可以后面再做。它比 mHC 更适合作为第一个研究分支。[AttnRes 论文](https://arxiv.org/abs/2603.15031)、[官方仓库](https://github.com/MoonshotAI/Attention-Residuals)

DeepSeek 那个准确名称是 mHC，即 Manifold-Constrained Hyper-Connections。它把 residual stream 扩展成多条并行流，再通过 Sinkhorn 投影约束动态混合矩阵。论文中的效率依赖融合内核、重计算和并行系统；直接用普通 PyTorch 实现，会显著增加残差激活和内存带宽，而长上下文本来就缺显存。因此可以实现，但不适合第一轮作为默认结构。[mHC 论文](https://arxiv.org/abs/2512.24880)

AttnRes 和 mHC 应当做成互斥实验项，分别与标准残差进行同预算对照，不建议一开始叠加，否则很难判断收益来自哪里。

六、MoE 当前不值得优先做。它解决的是总参数容量与激活计算量之间的关系，不解决上下文长度、注意力 O(T²) 或 KV cache。当前模型规模和单机训练条件下，router、负载均衡、专家分发以及额外优化器状态很可能让训练更慢。等 dense 长上下文基线稳定后，再把 SwiGLU 替换成小型 top-2 MoE 做研究更合理。

另外还有三个比 MoE 更值得先处理的结构细节：当前没有统一的权重初始化；SwiGLU 的 `d_ff=2048=4*d_model`，使 FFN 占每个 block 约 75% 参数，属于偏宽配置而不是常见的等参数 SwiGLU；embedding 和 lm_head 没有绑权，额外占约 419 万参数。这些不是必然错误，但重构时应该明确决定，而不是继续依赖默认值。

我的建议顺序是：先做 2K/4K dense 长上下文基线，整理共享 RoPE、GQA、静态 KV cache和明确初始化；训练侧接上 BF16、Flash 实际派发、梯度累积与 activation checkpointing；基线稳定后先实验 AttnRes，再单独实验 mHC；MLA、MoE、Kimi Linear 暂时放到后面的独立分支。这样每次结构变化都能测出真实收益，也不会把长上下文、推理缓存和研究型残差混成一团。
