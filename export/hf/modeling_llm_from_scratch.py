import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_llm_from_scratch import LLMFromScratchConfig
    from .model import Transformer as CoreTransformer
except ImportError:
    from export.hf.configuration_llm_from_scratch import LLMFromScratchConfig
    from models.model import Transformer as CoreTransformer

IGNORE_INDEX = -100


class LLMFromScratchForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LLMFromScratchConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _tied_weights_keys = {"lm_head.weight": "embedding.weight"}
    create_static_kv_cache = CoreTransformer.create_static_kv_cache

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # 核心模型维护自己的静态 cache，不接收 Transformers Cache 类型。
        return False

    def __init__(self, config: LLMFromScratchConfig):
        super().__init__(config)
        core = CoreTransformer(
            d_model=config.d_model,
            n_head=config.n_head,
            d_ff=config.d_ff,
            theta=config.theta,
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            tie_word_embeddings=config.tie_word_embeddings,
        )
        self.rope = core.rope
        self.layers = core.layers
        self.norm = core.norm
        self.context_length = core.context_length
        self.embedding = core.embedding
        self.lm_head = core.lm_head
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):
        if module is self.rope:
            # HF 从 meta 设备加载时必须重建未写入 state_dict 的 RoPE 表。
            frequency_indices = torch.arange(
                module.d_k // 2, dtype=torch.float64, device=module.inv_freq.device
            )
            module.inv_freq = (1.0 / module.theta ** (2.0 * frequency_indices / module.d_k)).float()
            module.cos_cached = torch.empty(0, module.d_k // 2, device=module.inv_freq.device)
            module.sin_cached = torch.empty_like(module.cos_cached)
            module._maybe_extend_cache(self.context_length, module.inv_freq.device)
        else:
            super()._init_weights(module)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")

        if use_cache is None:
            use_cache = self.config.use_cache

        logits, past_key_values = CoreTransformer.forward(
            self,
            input_ids,
            token_positions=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        if return_dict is False:
            output = (logits, past_key_values)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=True,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids")
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        if past_key_values is None and use_cache:
            past_key_values = CoreTransformer.create_static_kv_cache(
                self,
                batch_size=input_ids.size(0),
                max_cache_length=self.context_length,
            )
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
        }
