import json

# 读取你训练好的 tokenizer.json
with open("outputs/qwen_style_tokenizer.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 把 merges 从 ["a","b"] 变成 "a b" 格式
data["model"]["merges"] = [" ".join(pair) for pair in data["model"]["merges"]]

# 保存回去（覆盖原文件）
with open("outputs/qwen_style_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 已转换成千问官方 merges 格式！")