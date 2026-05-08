# infrence from trained model
from run_train_model import generate
import torch
import torch.nn.functional as F
from model import Transformer as Model
from tokenizer_optimized import Tokenizer

model_ckpt = r"../train_logs\run_20260508_194017\ckpt_iter_49999.pt"
vocab_file = "../bpe/tokenizer"
merge_file = "../bpe/tokenizer/qwen_style_tokenizer.json"


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"using device: {device}")

# init Tokenizer
tokenizer = Tokenizer(vocab_file)

# reconstruct model
model_args = dict(
    vocab_size=8192, 
    context_length=128, 
    n_head=8, 
    num_layers=12, 
    d_model=512, 
    d_ff=2048,
    theta=10000.0
)

print(f"Loading model from {model_ckpt}...")
model = Model(**model_args).to(device)

# load weights
checkpoint = torch.load(model_ckpt, map_location=device)
state_dict = checkpoint['model'] # it has keys: "model", "optimizer", "iteration".
model.load_state_dict(state_dict)
model.eval()

# 开始调戏模型
prompts = [
    "中国的首都是",
    "自然语言处理",
    "北京是一座",
    "深度学习是",
    'hello ,i am',
    '今天天气',
    '我今天吃了',
    '打个胶先，',
    '你是一个可爱的小傻逼，回答问题：你是傻逼吗？答：',
    '上路被三人越塔，',
    '乌兹，永远的'
]

print("-" * 30)
for p in prompts:
    print(f"Prompt: {p}")
    full_output, _ = generate(model, 
                              tokenizer=tokenizer, 
                              context=p, 
                              max_new_tokens=50, 
                              temperature=0.6, 
                              top_p=0.9, 
                              eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"),
                              context_length=128, 
                              device=device)
    print(f"Generated: {full_output}")
    print("=" * 80)