from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig
from tokenizer_optimized import Tokenizer # 用你原本写好的高效分词器
import torch

# 1. 加载
tokenizer = Tokenizer("../bpe/tokenizer")
eos_id = tokenizer.special_token_to_id.get("<|endoftext|>", 0)

model = AutoModelForCausalLM.from_pretrained("./my_sft_complete_model", trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
test_prompt = (
    "以下是单项选择题，请选出正确答案。\n"
    "题目：中国的首都是哪里？\n"
    "A. 上海\n"
    "B. 北京\n"
    "C. 广州\n"
    "D. 深圳\n"
    "答案："
)
# 2. 编码输入
inputs = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long, device=model.device)

# 3. 玩点高级配置：比如核采样 + 重复词惩罚 + 长度限制
gen_config = GenerationConfig(
    max_new_tokens=64,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.1, # 惩罚重复词，你原本的代码里可没有这个，现在直接白嫖！
    eos_token_id=eos_id,
    pad_token_id=eos_id,
)

# 4. 直接使用自带的 generate
with torch.no_grad():
    outputs = model.generate(inputs, generation_config=gen_config)

# 5. 解码输出
print(tokenizer.decode(outputs[0].tolist()))