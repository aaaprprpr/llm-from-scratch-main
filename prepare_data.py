import numpy as np
from tqdm import tqdm
from main.tokenizer_optimized import Tokenizer  # 注意路径和你训练脚本一致

# ===================== 固定配置 =====================
# 千问词表路径（你下载的那个 tokenizer.json）
VOCAB_PATH = "trained_tokenizer/tokenizer.json"

# 千问所有特殊 token（必须和训练代码里一致！）
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|file_sep|>"
]

# 千问固定 EOS token ID（从你给的词表看到的）
EOS_TOKEN_ID = 151643  # <|endoftext|>
# 你之前写的151664是 <|file_sep|>，不是结束符！！！

# ===================== 初始化分词器 =====================
print("正在加载千问分词器...")
tokenizer = Tokenizer.from_files(
    vocab_filepath=VOCAB_PATH,
    merges_filepath=VOCAB_PATH,  # 词表和合并规则都在同一个json里
    special_tokens=SPECIAL_TOKENS
)

# ===================== 生成 bin 文件 =====================
def build_bin(input_txt: str, output_bin: str):
    # 直接打开bin文件流式写入，不把所有token放内存里！！
    with open(output_bin, "wb") as f_out:
        print(f"正在处理：{input_txt} -> {output_bin}")
        with open(input_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc=f"生成 {output_bin}"):
            line = line.strip()
            if not line:
                continue
            tokens = tokenizer.encode(line)
            tokens.append(EOS_TOKEN_ID)
            
            # 流式写入，每次写一小段，不爆内存！
            arr = np.array(tokens, dtype=np.uint32)
            arr.tofile(f_out)
    
    print(f"✅ {output_bin} 完成！")

# ===================== 执行生成 =====================
if __name__ == "__main__":
    build_bin("val_raw.txt", "val.bin")
    build_bin("train_raw.txt", "train.bin")    
    print("\n🎉 全部完成！可以直接启动训练！")