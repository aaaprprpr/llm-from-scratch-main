from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import  ByteLevel, Split, Sequence
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers.normalizers import NFC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from glob import glob
import os
# ===================== 1. 初始化 BPE 分词器 =====================
tokenizer = Tokenizer(BPE(unk_token=None))  # 没有 unk token
os.makedirs("outputs", exist_ok=True)
# 预分词：字节级
qwen_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

tokenizer.pre_tokenizer = Sequence([
    Split(pattern=qwen_pattern, behavior="isolated"),
    ByteLevel(add_prefix_space=False, use_regex=False)
])
tokenizer.normalizer = NFC()
tokenizer.decoder = ByteLevelDecoder(add_prefix_space=False)
# tokenizer.post_processor = ByteLevelProcessor(add_prefix_space=False, use_regex=False)
# ===================== 2. 训练配置 =====================
trainer = BpeTrainer(
    # 词表大小：base 是 151851，你可以自己设
    vocab_size=65536,
    min_frequency=10,
    # 特殊 token
    special_tokens=[
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
    ],
    # 支持中文/全字符
    show_progress=True,
    initial_alphabet=ByteLevel.alphabet(), 
    continuing_subword_prefix="",
    num_workers=16,
    limit_alphabet=500,
    max_token_length=20,
)

# ===================== 3. 开始训练 =====================
# 你的语料文件
files = glob("data/*.txt")
# files = ["../data/clean_traditional_wiki.txt"]
tokenizer.train(
    files=files, 
    trainer=trainer,
)

# ===================== 4. 保存词表 =====================
# 保存格式：vocab.json + merges.txt
tokenizer.save("outputs/qwen_style_tokenizer.json")
# 也可以单独导出 vocab.json
# tokenizer.model.save("outputs")  # 会生成 vocab.json + merges.txt

# ===================== 5. 封装成 Transformers 可用格式 =====================
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    unk_token=None,
    pad_token="<|endoftext|>",
    # 特殊 token ID
    model_input_names=["input_ids", "attention_mask"]
)

# 保存成可以直接加载的分词器文件夹
wrapped_tokenizer.save_pretrained("./my_qwen_tokenizer")
print("训练完成！")
print("词表大小：", wrapped_tokenizer.vocab_size)

# 加载自己训练的分词器
tokenizer = AutoTokenizer.from_pretrained("./my_qwen_tokenizer")
# 测试分词
text = "你好，这是我自己训练的分词器！"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
print("分词：", tokens)
print("ID：", ids)
print("还原：", tokenizer.decode(ids))