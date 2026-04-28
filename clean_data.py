from datasets import load_dataset
import ftfy
import glob
import os

if __name__ == "__main__":
    # ================= 配置 =================
    GLOB_PATTERN = r"data/simplified/*.txt"  # 批量输入（改这里）
    OUTPUT_DIR = r"data/fixed"              # 输出文件夹（自动创建）
    NUM_THREADS = 16                         # 线程数不变
    # ========================================

    # 自动创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有文件
    file_list = glob.glob(GLOB_PATTERN)

    # 批量处理
    for RAW_PATH in file_list:
        # 输出文件名保持不变，只换文件夹
        filename = os.path.basename(RAW_PATH)
        SAVE_PATH = os.path.join(OUTPUT_DIR, filename)

        print(f"\n正在处理：{RAW_PATH}")
        print(f"输出到：{SAVE_PATH}")

        # 加载数据集
        ds = load_dataset("text", data_files=RAW_PATH, encoding="utf-8")

        # 清洗函数
        def clean_fn(row):
            text = ftfy.fix_text(row["text"])
            text = text.strip()
            return {"text": text}

        # 多线程并行处理
        ds = ds.map(
            clean_fn,
            num_proc=NUM_THREADS,
            desc="清洗文本"
        )

        # 过滤空行、过短行（≤5 字符丢掉）
        ds = ds.filter(
            lambda x: len(x["text"]) > 5,
            num_proc=NUM_THREADS,
            desc="过滤无效行"
        )

        # 高速写入
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(ds["train"]["text"]) + "\n")

        print(f"✅ 处理完成：{filename}")

    print("\n🎉 所有文件 ftfy 清洗完毕！")