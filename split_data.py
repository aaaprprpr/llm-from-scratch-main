import random

with open("data/clean_traditional_wiki.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
random.shuffle(lines)

# 95% 训练，5% 验证
split = int(0.95 * len(lines))
train = lines[:split]
val = lines[split:]

with open("data/train_raw.txt", "w", encoding="utf-8") as f: f.writelines(train)
with open("data/val_raw.txt", "w", encoding="utf-8") as f: f.writelines(val)