import numpy as np
from tqdm import tqdm
from main.tokenizer_optimized import Tokenizer
from concurrent.futures import ProcessPoolExecutor
import os

# ===================== 固定配置 =====================
VOCAB_PATH = "bpe/tokenizer"
EOS_TOKEN_ID = 0 
CHUNK_LINES = 2000  # 每个子进程一次处理的行数（调大这个可以减少IPC开销）

def init_process():
    """每个子进程初始化一次分词器"""
    global tokenizer
    tokenizer = Tokenizer(VOCAB_PATH)

def process_chunk(lines):
    """
    子进程逻辑：处理一整块文本，返回一个合并后的 NumPy 数组。
    减少主进程写入次数，实现“写优先”。
    """
    global tokenizer
    all_tokens = []
    for line in lines:
        line = line.strip()
        if line:
            tokens = tokenizer.encode(line)
            tokens.append(EOS_TOKEN_ID)
            all_tokens.extend(tokens)
    
    if not all_tokens:
        return None
    # 词表 < 65536 建议用 uint16，否则用 uint32
    return np.array(all_tokens, dtype=np.uint32)

def line_batch_generator(file_path, batch_size):
    """生成器：按块读取文件内容"""
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def build_bin(input_txt: str, output_bin: str):
    # 1. 预估文件大小（用于进度条）
    total_size = os.path.getsize(input_txt)
    
    print(f"\n🚀 开始多进程分块处理：{input_txt}")
    
    # 2. 准备写入
    with open(output_bin, "wb") as f_out:
        # 使用 max_workers=cpu_count()-1 留一个核心给操作系统调度IO
        max_workers = max(1, os.cpu_count() - 1)
        
        with ProcessPoolExecutor(max_workers=6, initializer=init_process) as executor:
            # 建立任务生成器
            batches = line_batch_generator(input_txt, CHUNK_LINES)
            
            # 使用 tqdm 监控进度（基于文件字节数更准确）
            pbar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"生成 {output_bin}")
            
            # map 会保持顺序返回结果
            for result_array in executor.map(process_chunk, batches):
                if result_array is not None:
                    # 关键：大块数据直接 dump 到磁盘，这是最高效的写方式
                    result_array.tofile(f_out)
                    
                    # 估算更新进度条：由于 chunk 处理，按字节更新比较难精确
                    # 我们这里近似更新：每批次处理完大约消耗的字节数
                    # 简单点可以根据 input_txt 的处理行数比例来做，
                    # 这里直接用一个 batch 的大致字节数更新：
                    pbar.update(CHUNK_LINES * 150) # 假设平均每行 150 字节

            # 确保最后进度条填满
            pbar.n = total_size
            pbar.refresh()
            pbar.close()

    print(f"✅ {output_bin} 处理完成！")

if __name__ == "__main__":
    build_bin("data/val.txt", "data/val.bin")
    build_bin("data/train.txt", "data/train.bin")
