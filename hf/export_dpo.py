import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or Path.cwd()).resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from config_loader import Config
from hf.configuration_llm_from_scratch import LLMFromScratchConfig
from hf.modeling_llm_from_scratch import LLMFromScratchForCausalLM

CONFIG_PATH = PROJECT_ROOT / "configs" / "dpo.json"
CONFIG_CODE_PATH = PROJECT_ROOT / "hf" / "configuration_llm_from_scratch.py"
MODEL_CODE_PATH = PROJECT_ROOT / "hf" / "modeling_llm_from_scratch.py"
CORE_MODEL_CODE_PATH = PROJECT_ROOT / "models" / "model.py"
CORE_MODEL_MODULES_PATH = PROJECT_ROOT / "models" / "modules"
CORE_MODEL_UTILS_PATH = PROJECT_ROOT / "models" / "utils.py"


def load_config() -> Config:
    return Config(CONFIG_PATH)


def build_hf_config(config: Config, tokenizer) -> LLMFromScratchConfig:
    model_config = config.require("model")
    hf_config = LLMFromScratchConfig(
        **model_config,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    hf_config.architectures = ["LLMFromScratchForCausalLM"]
    hf_config.auto_map = {
        "AutoConfig": "configuration_llm_from_scratch.LLMFromScratchConfig",
        "AutoModel": "modeling_llm_from_scratch.LLMFromScratchForCausalLM",
        "AutoModelForCausalLM": ("modeling_llm_from_scratch.LLMFromScratchForCausalLM"),
    }
    return hf_config


def load_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.resolve_path("paths", "tokenizer"))
    tokenizer.chat_template = config.resolve_path("paths", "chat_template").read_text(
        encoding="utf-8"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def copy_remote_code_files(output_dir: Path):
    shutil.copy2(CONFIG_CODE_PATH, output_dir / "configuration_llm_from_scratch.py")
    shutil.copy2(MODEL_CODE_PATH, output_dir / "modeling_llm_from_scratch.py")
    shutil.copy2(CORE_MODEL_CODE_PATH, output_dir / "model.py")
    shutil.copy2(CORE_MODEL_MODULES_PATH / "__init__.py", output_dir / "modules.py")
    for module_name in ("rope.py", "attention.py", "feed_forward.py", "block.py"):
        shutil.copy2(CORE_MODEL_MODULES_PATH / module_name, output_dir / module_name)
    shutil.copy2(CORE_MODEL_UTILS_PATH, output_dir / "utils.py")


def write_readme(output_dir: Path):
    readme = """# LLM From Scratch DPO Model

This directory is a HuggingFace-style export of the local DPO model.

Transformers usage:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "hf/dpo_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
```

vLLM usage may require the Transformers modeling backend:

```powershell
vllm serve hf/dpo_model --task generate --model-impl transformers --trust-remote-code
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main():
    config = load_config()
    weights_path = config.resolve_path("paths", "clean_weights")
    output_dir = config.resolve_path("paths", "hf_export")

    print(f"加载 DPO 纯权重：{weights_path}")
    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    tokenizer = load_tokenizer(config)
    hf_config = build_hf_config(config, tokenizer)
    model = LLMFromScratchForCausalLM(hf_config)
    model.load_state_dict(state_dict, strict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    copy_remote_code_files(output_dir)
    write_readme(output_dir)

    print(f"HF 目录已保存到：{output_dir}")
    print("Transformers 加载时需要 trust_remote_code=True")


if __name__ == "__main__":
    main()
