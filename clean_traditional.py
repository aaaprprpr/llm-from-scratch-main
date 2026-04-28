from opencc import OpenCC
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import glob
import os

# 全局单例转换器（进程内唯一）
cc = OpenCC("t2s")

def has_traditional(text: str) -> bool:
    text = text.strip()
    if not re.search(r"[\u4e00-\u9fff]", text):
        return False
    return text != cc.convert(text)

# 单行处理函数
def filter_line(line: str) -> str | None:
    if not has_traditional(line):
        return line
    return None

# ========== 配置 ==========
GLOB_PATTERN = "data/clean/*.txt"  # 你要批量处理的文件
OUTPUT_DIR = "data/simplified"     # 输出：简体文件夹
BATCH_SIZE = 10000
PROC_NUM = cpu_count()
# ==========================

if __name__ == "__main__":
    # 自动创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有要处理的文件列表
    file_list = glob.glob(GLOB_PATTERN)
    
    for file_path in file_list:
        keep_count = 0
        filename = os.path.basename(file_path)
        OUT_PATH = os.path.join(OUTPUT_DIR, filename)
        
        print(f"\n正在处理：{file_path}")

        # 统计总行数
        with open(file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        # 读取全部行
        print("加载文本中...")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 多进程并行过滤
        print(f"多进程启动，进程数：{PROC_NUM}")
        with Pool(processes=PROC_NUM) as pool, tqdm(total=total_lines, desc="繁体过滤", unit="行") as pbar:
            res_iter = pool.imap(filter_line, lines, chunksize=BATCH_SIZE)
            valid_lines = []
            for res in res_iter:
                pbar.update(1)
                if res is not None:
                    valid_lines.append(res)
                    keep_count += 1

        # 写入新文件夹，文件名不变
        with open(OUT_PATH, "w", encoding="utf-8") as f_out:
            f_out.writelines(valid_lines)

        print(f"\n✅ 处理完成：{filename}")
        print(f"原始总行数：{total_lines}")
        print(f"保留简体行数：{keep_count}")

