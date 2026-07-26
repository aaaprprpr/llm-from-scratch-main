# vLLM

The exported DPO model lives in `hf/dpo_model`.

Run vLLM from Python:

```bash
python vllm/serve_dpo.py
```

The model uses a custom HuggingFace architecture, so `--trust-remote-code` is required.
The Python script uses vLLM's `LLM` API with `model_impl="transformers"`.
