import torch
from hf_wrapper import MyCustomLLMConfig, MyCustomLLMForCausalLM
MyCustomLLMConfig.register_for_auto_class()
MyCustomLLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")
# 1. 定义你的模型参数（请务必与你预训练时的参数完全一致！）
config = MyCustomLLMConfig(
    d_model = 512,
    n_head = 8,
    d_ff = 2048,
    theta = 10000.0,
    vocab_size = 8192,
    context_length = 256,
    num_layers = 12,
)

# 2. 初始化 HF 格式的模型（此时内部参数是随机的）
hf_model = MyCustomLLMForCausalLM(config)

# 3. 读取你原本的 .pt 预训练权重
# 假设你的 pt 文件里保存的就是状态字典。如果是完整的 model 对象，请使用 torch.load().state_dict()
checkpoint_path = "../train_logs/run_20260508_194017/ckpt_iter_49999.pt" # 换成你真实的路径
print(f"正在从 {checkpoint_path} 读取权重...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
raw_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

# 【核心对齐】因为我们套了一层 self.transformer，所以权重名字前面会自动多出一个 "transformer." 前缀
# 我们需要把你的原始权重 Key 自动映射上去
hf_state_dict = {}
for key, value in raw_state_dict.items():
    hf_state_dict[f"transformer.{key}"] = value

# 4. 将对齐后的权重安全灌入
hf_model.load_state_dict(hf_state_dict)
print("权重成功无缝对接！")

# 5. 保存为标准的 Hugging Face 模型目录
# 这会在 ./my_hf_model 文件夹下生成 config.json 和 pytorch_model.bin（或 safetensors）
hf_model.save_pretrained("./my_hf_model")