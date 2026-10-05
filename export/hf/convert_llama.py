"""将本项目 dense Transformer 的权重等价映射到标准 Llama。"""
from transformers import LlamaConfig, LlamaForCausalLM


def build_llama_config(model_args: dict, tokenizer=None) -> LlamaConfig:
    token_ids = {}
    if tokenizer is not None:
        end_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
        token_ids = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": list(dict.fromkeys(end_ids)),
            "pad_token_id": tokenizer.pad_token_id,
        }
    return LlamaConfig(
        vocab_size=model_args["vocab_size"],
        hidden_size=model_args["d_model"],
        intermediate_size=model_args["d_ff"],
        num_hidden_layers=model_args["num_layers"],
        num_attention_heads=model_args["n_head"],
        num_key_value_heads=model_args["n_head"],
        max_position_embeddings=model_args["context_length"],
        rope_theta=model_args["theta"],
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=model_args.get("tie_word_embeddings", False),
        **token_ids,
    )


def reorder_rope_rows(weight, n_head: int):
    # 核心 RoPE 旋转 (0,1)、(2,3)，Llama 旋转 (0,D/2)、(1,D/2+1)。
    # 每个头把投影输出从 [e0,o0,e1,o1,...] 重排为 [e0,e1,...,o0,o1,...]。
    hidden_size = weight.shape[0]
    head_dim = hidden_size // n_head
    return weight.reshape(n_head, head_dim // 2, 2, -1).transpose(1, 2).reshape_as(weight).contiguous()


def convert_state_dict(state_dict: dict, model_args: dict) -> dict:
    result = {
        "model.embed_tokens.weight": state_dict["embedding.weight"],
        "model.norm.weight": state_dict["norm.weight"],
        "lm_head.weight": state_dict["lm_head.weight"],
    }
    for index in range(model_args["num_layers"]):
        source = f"layers.{index}."
        target = f"model.layers.{index}."
        q, k, v = state_dict[source + "attn.qkv_proj.weight"].chunk(3, dim=0)
        result[target + "self_attn.q_proj.weight"] = reorder_rope_rows(q, model_args["n_head"])
        result[target + "self_attn.k_proj.weight"] = reorder_rope_rows(k, model_args["n_head"])
        result[target + "self_attn.v_proj.weight"] = v.contiguous()
        for old, new in (
            ("attn.out_proj.weight", "self_attn.o_proj.weight"),
            ("attn_norm.weight", "input_layernorm.weight"),
            ("ffn_norm.weight", "post_attention_layernorm.weight"),
            ("ffn.w1.weight", "mlp.gate_proj.weight"),
            ("ffn.w3.weight", "mlp.up_proj.weight"),
            ("ffn.w2.weight", "mlp.down_proj.weight"),
        ):
            result[target + new] = state_dict[source + old]
    return result


def build_llama_model(checkpoint: dict, tokenizer=None):
    model_args = checkpoint["model_args"]
    model = LlamaForCausalLM(build_llama_config(model_args, tokenizer))
    model.load_state_dict(convert_state_dict(checkpoint["model"], model_args), strict=True)
    return model.eval()
