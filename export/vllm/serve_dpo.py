import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.dpo.utils import load_config


PROMPTS = [
    "中国的首都是哪里？",
    "自然语言处理是什么？",
    "请用三句话介绍深度学习。",
]


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def clean_answer(text: str) -> str:
    text = text.split("<|im_end|>", 1)[0]
    text = text.split("<|endoftext|>", 1)[0]
    return text.strip()


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="用 vLLM 原生 Llama 后端离线生成")
    parser.add_argument("--model", type=Path, default=config.resolve_path("paths", "hf_export"))
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model),
        runner="generate",
        model_impl="vllm",
    )
    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt(tokenizer, prompt) for prompt in PROMPTS]

    sampling_params = SamplingParams(
        max_tokens=80,
        temperature=0.7,
        top_p=0.9,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(PROMPTS, outputs):
        answer = clean_answer(output.outputs[0].text)
        print(f"用户：{prompt}")
        print(f"助手：{answer}")
        print("=" * 80)


if __name__ == "__main__":
    main()
