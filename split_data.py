import random
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= 配置 =================
GLOB_PATTERN = "data/simplified/*.txt"
TRAIN_PATH = "data/train.txt"
VAL_PATH =   "data/val.txt"
TRAIN_RATIO = 0.9
BATCH_SIZE = 20000  # 每批 2 万行（内存极低）
PROC_NUM = cpu_count()
# ========================================

# 子进程里：处理一个批次 → 随机 → 划分
def process_batch(batch):
    random.shuffle(batch)
    split_idx = int(len(batch) * TRAIN_RATIO)
    return batch[:split_idx], batch[split_idx:]

if __name__ == "__main__":
    file_list = glob.glob(GLOB_PATTERN)
    print(f"找到文件数：{len(file_list)}")

    with open(TRAIN_PATH, "w", encoding="utf-8") as f_train, \
         open(VAL_PATH, "w", encoding="utf-8") as f_val, \
         Pool(processes=PROC_NUM) as pool:

        for file in file_list:
            print(f"\n处理文件：{file}")
            
            with open(file, "r", encoding="utf-8") as f_in:
                current_batch = []
                pbar = tqdm(desc=f"处理中", unit="行")

                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    current_batch.append(line + "\n")

                    # 攒够一批 → 丢进多进程处理
                    if len(current_batch) >= BATCH_SIZE:
                        # 异步进程池处理
                        train_part, val_part = pool.apply(process_batch, args=(current_batch,))
                        
                        f_train.writelines(train_part)
                        f_val.writelines(val_part)
                        
                        current_batch = []
                        pbar.update(BATCH_SIZE)

                # 最后一批
                if current_batch:
                    train_part, val_part = process_batch(current_batch)
                    f_train.writelines(train_part)
                    f_val.writelines(val_part)
                    pbar.update(len(current_batch))

                pbar.close()

    print("\n✅ 多进程低内存划分完成！")