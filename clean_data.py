from datasets import load_dataset
import ftfy


if __name__ == "__main__":


    # ================= 配置 =================
    RAW_PATH = r"data\wiki_cn_clean.txt"    # 你的输入文件
    SAVE_PATH = r"cleaned_wiki.txt"         # 输出文件
    NUM_THREADS = 8                         # 8线程加速
    # ========================================

    # 加载数据集
    ds = load_dataset("text", data_files=RAW_PATH, encoding="utf-8")

    # 清洗函数
    def clean_fn(row):
        text = ftfy.fix_text(row["text"])
        text = text.strip()
        return {"text": text}

    # ========== 核心：多线程并行处理 ==========
    ds = ds.map(
        clean_fn,
        num_proc=NUM_THREADS,  # 8线程
        desc="清洗文本"
    )

    # 过滤空行、过短行
    ds = ds.filter(
        lambda x: len(x["text"]) > 5,
        num_proc=NUM_THREADS,
        desc="过滤无效行"
    )

    # 导出干净数据
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        # 一次性 join 所有文本，系统级高速写入
        f.write("\n".join(ds["train"]["text"]) + "\n")

    print(f"✅ 清洗完成！已保存到：{SAVE_PATH}")