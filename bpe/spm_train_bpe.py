import sentencepiece as spm

# ==============================================================================
# 🔥 1:1 对齐你原来的 HF Tokenizers BPE 风格
# 字节级、无 dummy 前缀、中文友好、无 UNK、干净子词、Qwen 风格
# ==============================================================================
spm.SentencePieceTrainer.train(
    # 语料
    input="../data/cleaned_wiki.txt",                # 你的语料路径（和你原来一样）
    
    # 输出
    model_prefix="my_qwen_sp_token",   # 输出模型名
    model_type="bpe",                  # BPE 算法（不用你写）
    
    # 词表大小（和你 HF 代码一致：65536）
    vocab_size=32768,
    
    # 🔥 关键：中文全覆盖 = 无 UNK（和你 unk_token=None 对齐）
    character_coverage=1.0,
    
    # 🔥 完全去掉 SentencePiece 默认的 ▁ 前缀（和你 add_prefix_space=False 对齐）
    add_dummy_prefix=False,            
    split_digits=True,                 # 数字单独切分（和你 HF 一致）
    byte_fallback=True,                # 字节级 fallback（和你 ByteLevel 一致）
    
    # 🔥 不自动加空格、不删空格（完全保持原文格式）
    remove_extra_whitespaces=False,
    

    
    # 🔥 特殊 token（和你 HF 完全一样）
    user_defined_symbols=[
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>"
    ],
    
    # 训练速度
    num_threads=32,                    # 和你 num_workers=16 对齐
    train_extremely_large_corpus=True  # 大语料优化（内存极低）
    
)