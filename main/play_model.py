# infrence from trained model
from run_train_model import generate
import torch
import torch.nn.functional as F
from model import Transformer as Model
from tokenizer_optimized import Tokenizer

model_ckpt = "C:/Users/guojia/PycharmProjects/heima/llm-from-scratch-main/train_logs/run_20260425_011651/ckpt_iter_4199.pt"
vocab_file = "C:/Users/guojia/PycharmProjects/heima/llm-from-scratch-main/trained_tokenizer/tokenizer.json"
merge_file = "C:/Users/guojia/PycharmProjects/heima/llm-from-scratch-main/trained_tokenizer/tokenizer.json"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# init Tokenizer
tokenizer = Tokenizer.from_files(vocab_file, merge_file, special_tokens=["<|endoftext|>"])
eos_id = tokenizer.special_token_to_id.get("<|endoftext|>")

# reconstruct model
model_args = dict(
    vocab_size=151665, 
    context_length=128, 
    n_head=16, 
    num_layers=4, 
    d_model=512, 
    d_ff=1344,
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
]

print("-" * 30)
for p in prompts:
    print(f"Prompt: {p}")
    full_output, _ = generate(model, 
                              tokenizer=tokenizer, 
                              context=p, 
                              max_new_tokens=50, 
                              temperature=0.8, 
                              top_p=0.9, 
                              eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"), 
                              context_length=128, 
                              device=device)
    print(f"Generated: {full_output}")
    print("=" * 80)