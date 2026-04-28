from datasets import load_dataset
from tqdm import tqdm  # 进度条

# 自动下载 + 自动合并 6 个 parquet 文件
print("正在加载数据集...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")

# 导出成纯文本 + 显示进度条
print("开始导出为 wiki_cn_clean.txt ...")
with open("wiki_cn_clean.txt", "w", encoding="utf-8") as f:
    for item in tqdm(dataset, desc="导出文本进度"):
        f.write(item["text"] + "\n")

print("✅ 导出完成！文件：wiki_cn_clean.txt")




