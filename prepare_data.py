import numpy as np
from tqdm import tqdm
from main.tokenizer_optimized import Tokenizer  # 注意路径和你训练脚本一致
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
# ===================== 固定配置 =====================
VOCAB_PATH = "bpe/tokenizer"

# 所有特殊 token（必须和训练代码里一致！）
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]

# 千问固定 EOS token ID（从你给的词表看到的）
EOS_TOKEN_ID = 0  # <|endoftext|>
# ===================== 初始化分词器 =====================
def init_process():
    """进程启动时执行一次：全局创建tokenizer，不重复加载！"""
    global tokenizer
    tokenizer = Tokenizer(VOCAB_PATH)

# ===================== 多进程加速核心 =====================
def process_line(line):
    """单个行处理函数（给多进程调用）"""
    line = line.strip()
    if not line:
        return None
    tokens = tokenizer.encode(line)
    tokens.append(EOS_TOKEN_ID)
    return np.array(tokens, dtype=np.uint32)


# ===================== 生成 bin 文件 =====================
def build_bin(input_txt: str, output_bin: str):
    # 直接打开bin文件流式写入，不把所有token放内存里！！
    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"\n开始处理：{input_txt}")
    print(f"总行数：{len(lines):,}")


    with open(output_bin, "wb") as f_out:
        # 3. 多进程并行分词（吃满所有CPU）
        with ProcessPoolExecutor(max_workers=12,initializer=init_process) as executor:
            # 流式生成结果，处理完一个就写一个，不堆积在内存！
            for result in tqdm(
                executor.map(process_line, lines, chunksize=50),
                total=len(lines),
                desc=f"生成 {output_bin}",
                mininterval=0.5,
                miniters=1000
            ):
                if result is not None:
                    result.tofile(f_out)
    
    print(f"✅ {output_bin} 完成！")

# ===================== 执行生成 =====================
if __name__ == "__main__":
    build_bin("data/val.txt", "data/val.bin")
    build_bin("data/train.txt", "data/train.bin")    
    print("\n🎉 全部完成！可以直接启动训练！")