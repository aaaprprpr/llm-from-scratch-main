# vLLM 推理

先按 [HF 导出说明](../hf/README.md)生成标准 Llama 目录，再在有 vLLM 的算力环境运行：

```bash
python -m export.vllm.serve_dpo
python -m export.vllm.serve_dpo --model export/hf/sft_model
```

入口是离线生成示例，默认目录来自 `configs/dpo.json` 的 `paths.hf_export`。它使用 vLLM 原生 Llama 实现与导出目录里的分词器、对话模板，不需要 `trust_remote_code`。后端选择见 [vLLM 官方参数说明](https://docs.vllm.ai/en/stable/api/vllm/config/model/)。

需要 HTTP 聊天接口时，可以在算力机自行运行：

```bash
vllm serve export/hf/dpo_model --served-model-name scratch-chat
```

本次只验证了权重映射、Transformers 前向和 KV cache；本机没有安装 vLLM，没有启动 GPU 推理或服务。旧自定义 HF 目录需重新导出为标准 Llama，不能直接套用这个入口。
