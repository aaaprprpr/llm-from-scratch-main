import torch
from model import Transformer as Model
from tokenizer_optimized import Tokenizer

model_ckpt = r"../train_logs\run_20260706_175441\ckpt_step_130000.pt"
vocab_file = "../bpe/tokenizer"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
print(f"Loading model from {model_ckpt}...")

tokenizer = Tokenizer(vocab_file)
model_args = dict(
    vocab_size=8192,
    context_length=256,
    n_head=8,
    num_layers=12,
    d_model=512,
    d_ff=2048,
    theta=10000.0,
)
model = Model(**model_args).to(device)
checkpoint = torch.load(
    model_ckpt,
    map_location="cpu",
    weights_only=True,
    mmap=True,
)
model.load_state_dict(
    checkpoint["model"],
    strict=True,
)
del checkpoint

model = model.to(device)
model.eval()

# 开始调戏模型
prompts = [
    "中国的首都是",
    "自然语言处理",
    "北京是一座",
    "深度学习是",
    "hello ,i am",
    "今天天气",
    "我今天吃了",
    "打个胶先，",
    "你是一个可爱的小傻逼，回答问题：你是傻逼吗？答：",
    "上路被三人越塔，",
    "乌兹，永远的",
]

print("-" * 30)
for p in prompts:
    print(f"Prompt: {p}")
    idx = tokenizer.idx(p, device=device)
    full_output = model.generate(
        idx,
        max_new_tokens=50,
        temperature=0.6,
        top_p=0.9,
        eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"),
        context_length=128,
    )
    print(f"Generated: {tokenizer.text(full_output,device=device)}")
    print("=" * 80)
