import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from model import Transformer

# 1. 定义符合 HF 规范的配置类
class MyCustomLLMConfig(PretrainedConfig):
    model_type = "my_custom_llm"
    attribute_map = {
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "n_head",
        "hidden_size": "d_model"
    }
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        d_ff: int = 2048,
        theta: float = 10000.0,
        vocab_size: int = 8192,
        context_length: int = 256,
        num_layers: int = 12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff
        self.theta = theta
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

# 2. 定义符合 HF 规范的模型包装类
class MyCustomLLMForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = MyCustomLLMConfig
    main_input_name = "input_ids"
    def __init__(self, config: MyCustomLLMConfig):
        super().__init__(config)
        self.transformer = Transformer(
            d_model=config.d_model,
            n_head=config.n_head,
            d_ff=config.d_ff,
            theta=config.theta,
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers
        )
        self._all_tied_weights_keys = []
    @property
    def all_tied_weights_keys(self):
        # 🟢 强力防御：直接给它返回一个空字典，通杀所有的源码版本
        return {}

    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor = None, 
        token_positions: torch.Tensor = None, 
        past_key_values = None,
        use_cache: bool = None,
        **kwargs
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if past_key_values is not None and not isinstance(past_key_values, (list, tuple)):
            # 情况 A：如果它已经是存了东西的 DynamicCache 实例
            if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
                legacy_past_key_values = []
                for layer_idx in range(len(past_key_values.key_cache)):
                    k = past_key_values.key_cache[layer_idx]
                    v = past_key_values.value_cache[layer_idx]
                    legacy_past_key_values.append([k, v])
                past_key_values = legacy_past_key_values
            else:
                # 情况 B：它是刚初始化、里面空空如也的 DynamicCache 壳子
                # 强行把它洗成 None，让底层触发完美的首次 Prefill 逻辑！
                past_key_values = None

        # 闭环：把全量缓存参数带给底层的普通模型
        logits, next_cache = self.transformer(
            x=input_ids,
            token_positions=token_positions,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))


        return_cache = None
        if next_cache is not None:
            # 看看外部给的是不是标准的 DynamicCache，如果是，把新层数据更新进去
            if "past_key_values" in kwargs and hasattr(kwargs["past_key_values"], "update"):
                return_cache = kwargs["past_key_values"]
                # 遍历底层吐出的每一层新的 [k, v]
                for layer_idx, (k, v) in enumerate(next_cache):
                    # 只有当前步新生成的那个 token 的 KV 需要被 update 进去
                    # 考虑到底层实现：如果不是第一次 prefill，底层往往只吐出增量词的 [:, :, -1:, :]
                    # 动态适配最新 token 维度
                    return_cache.update(k, v, layer_idx)
            else:
                # 回滚兼容旧的传统格式
                return_cache = next_cache
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=return_cache,
            hidden_states=None,
            attentions=None
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        完美对接 Hugging Face 全套 DynamicCache 和传统 Cache 混合模式的核心转换函数
        """
        position_ids = kwargs.get("position_ids", None)
        
        if past_key_values is not None:
            # 增量推理阶段（Decode）：只把当前最新生成的最后一个 token 送入模型
            input_ids = input_ids[:, -1:]
            
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            else:
                # 🔴 终极兼容修复：动态获取 KV cache 内部已经保存的序列长度
                if hasattr(past_key_values, "get_seq_length"):
                    # 针对标准库新版 DynamicCache 对象的原生兼容
                    past_length = past_key_values.get_seq_length()
                else:
                    # 针对你原本手写的传统列表/元组缓存 [layer][key/value] 的向前兼容
                    past_length = past_key_values[0][0].shape[-2]
                    
                position_ids = torch.tensor([[past_length]], dtype=torch.long, device=input_ids.device)

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        
        if position_ids is not None:
            model_inputs["token_positions"] = position_ids

        return model_inputs