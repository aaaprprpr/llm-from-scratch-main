from datasets import load_dataset
from tqdm import tqdm  # 进度条
import random
# 自动下载 + 自动合并 6 个 parquet 文件
print("正在加载数据集...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")

# 导出成纯文本 + 显示进度条
print("开始导出为 wiki_cn_clean.txt ...")
with open("wiki_cn_clean.txt", "w", encoding="utf-8") as f:
    for item in tqdm(dataset, desc="导出文本进度"):
        f.write(item["text"] + "\n")

print("✅ 导出完成！文件：wiki_cn_clean.txt")




with open("wiki_cn_clean.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
random.shuffle(lines)

# 95% 训练，5% 验证
split = int(0.95 * len(lines))
train = lines[:split]
val = lines[split:]

with open("train_raw.txt", "w", encoding="utf-8") as f: f.writelines(train)
with open("val_raw.txt", "w", encoding="utf-8") as f: f.writelines(val)