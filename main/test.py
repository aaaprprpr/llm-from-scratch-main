from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig
from tokenizer_optimized import Tokenizer 
import torch
import warnings

# 强制压制警告
warnings.filterwarnings("ignore", category=UserWarning)

# 1. 加载
tokenizer = Tokenizer("../bpe/tokenizer")
eos_id = tokenizer.special_token_to_id.get("<|endoftext|>", 0)

model = AutoModelForCausalLM.from_pretrained("./my_sft_complete_model", trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# ==================== 【唯一修改点：死死对齐你的 sft.py 模板】 ====================
instruction = "请写一段话夸赞春天的景色。"
# 格式和 sft.py 里的格式多一个空格、少一个换行都不行！必须一字不差！
test_prompt = f"以下是指令，请给出合适的回答。\n指令：{instruction}\n答案："
# ==================================================================================

# 2. 编码输入
inputs = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long, device=model.device)

# 3. 沿用你原本完全通过的高级配置
gen_config = GenerationConfig(
    max_new_tokens=128,        # 中文夸赞稍微放长一点到 128
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.1, 
    eos_token_id=eos_id,
    pad_token_id=eos_id,
    renormalize_logits=False 
)

# 4. 直接使用自带的 generate，不传 attention_mask
with torch.no_grad():
    outputs = model.generate(
        inputs, 
        generation_config=gen_config
    )

# 5. 解码输出
print("=== 模型输出 ===")
print(tokenizer.decode(outputs[0].tolist()))