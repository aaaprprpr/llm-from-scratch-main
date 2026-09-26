from transformers import PretrainedConfig


class LLMFromScratchConfig(PretrainedConfig):
    model_type = "llm_from_scratch"

    def __init__(
        self,
        vocab_size=8192,
        context_length=256,
        n_head=8,
        num_layers=12,
        d_model=512,
        d_ff=2048,
        theta=10000.0,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        **kwargs,
    ):
        use_cache = kwargs.pop("use_cache", False)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_head = n_head
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.theta = theta

        self.hidden_size = d_model
        self.intermediate_size = d_ff
        self.num_attention_heads = n_head
        self.num_hidden_layers = num_layers
        self.max_position_embeddings = context_length
