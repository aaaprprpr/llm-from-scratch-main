# -*- coding: utf-8 -*-
import re
import glob
from tqdm import tqdm
import os
# ====================== 正则规则（核心完全不变） ======================
CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]')
LINK_PATTERN = re.compile(r'^(https?://\S+|www\.\S+)$', re.IGNORECASE)
PURE_ENGLISH = re.compile(r'^[a-zA-Z\s\.,!?\';:-]+$')
GIBBERISH = re.compile(r'^[a-zA-Z0-9]+$')

# ====================== 清理函数（不变） ======================
def clean_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None

    # 最高优先级：有中文就整行保留
    if CHINESE_PATTERN.search(line):
        return line

    # 无中文 → 清理
    if LINK_PATTERN.match(line):
        return None
    if PURE_ENGLISH.match(line):
        return None
    if GIBBERISH.match(line):
        return None

    return None

# ====================== 批量处理（支持 glob） ======================
def clean_files(glob_pattern: str, output_dir: str):
    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    input_files = glob.glob(glob_pattern)
    
    for file in input_files:
        # 只取文件名，不要路径
        filename = os.path.basename(file)
        output_file = os.path.join(output_dir, filename)
        
        with open(file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            lines = f_in.readlines()
            for line in tqdm(lines, desc=f"处理：{filename}", unit="行"):
                cleaned = clean_line(line)
                if cleaned is not None:
                    f_out.write(cleaned + '\n')

# ====================== 运行 ======================
if __name__ == "__main__":
    # 在这里写你的通配符
    GLOB_PATTERN = "data/raw/*.txt"          # 清理当前目录所有 txt
    OUTPUT_PREFIX = "data/clean"
    clean_files(GLOB_PATTERN, OUTPUT_PREFIX)