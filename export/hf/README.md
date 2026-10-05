# Hugging Face 导出

默认把 DPO 精简权重转成标准 `LlamaForCausalLM`，同时保存 tokenizer 和 ChatML 模板：

```bash
python -m train.dpo.export
python -m export.hf.export_dpo
python -m export.hf.play_dpo_model
```

默认输入与输出分别由 `configs/dpo.json` 的 `paths.clean_weights`、`paths.hf_export` 指定。也可以直接指定含 `model`、`model_args` 的预训练、SFT 或 DPO checkpoint；SFT 示例：

```bash
python -m train.sft.export
python -m export.hf.export_dpo --checkpoint output/sft_weights/model.pt --output export/hf/sft_model
python -m export.hf.play_dpo_model --model export/hf/sft_model
```

精简导出目录附带的 `tokenizer/` 与 `chat_template.jinja` 优先使用；直接输入训练 checkpoint 时，分词器和模板来自 `configs/dpo.json`，需要与该权重匹配。旧版仅含裸 state_dict 的文件没有架构信息，请用原训练 checkpoint 重新导出。

转换保持原网络计算：拆开融合 QKV，并重排每个头的 Q/K 投影行，使本项目相邻维度 RoPE 与 Llama 的半区旋转一致；嵌入、RMSNorm、SwiGLU 和输出权重对应映射。词表、上下文长度及共享权重设置来自 checkpoint，不额外扩词表或扩窗口。

导出目录包含 `config.json`、safetensors 权重、分词器、聊天模板与 `training_metadata.json`。它可用 Transformers 的 `AutoModelForCausalLM.from_pretrained` 加载；[vLLM 入口](../vllm/README.md)使用同一目录。原 `configuration_llm_from_scratch.py` 和 `modeling_llm_from_scratch.py` 留作旧自定义格式兼容，新导出不复制它们。
