# import numpy as np
# import regex as re
# with open("C:/Users/guojia/PycharmProjects/heima/llm-from-scratch-main/wiki_cn_clean.txt", "r", encoding="utf-8") as f:
#     lines = f.readlines()
# print(len(lines))
# # for i in lines[:5]:
# #     print(i)
# PAT = r"(\s+|\S+)"
# RX=re.findall(PAT, lines[0])
# print(RX)

import os

if __name__ == '__main__':
    os.makedirs("bpe/data", exist_ok=True)
    target_size_mb = 100
    target_size = target_size_mb * 1024 * 1024  # 100MB
    PATH_OWT = "data/clean_traditional_wiki.txt"

    file_index = 1
    buffer = b""  # 内存缓存，保存上一次剩下的数据

    with open(PATH_OWT, "rb") as f_in:
        while True:
            # 读取新数据
            chunk = f_in.read(8192 * 1024)  # 每次读 8MB，稳定高效
            if not chunk:
                # 处理最后剩余的数据
                if buffer:
                    output_path = f"bpe/data/owt_chunk_{file_index}.txt"
                    with open(output_path, "wb") as f_out:
                        f_out.write(buffer)
                    print(f"已生成：{output_path}  大小：{len(buffer) / 1024 / 1024:.2f}MB")
                break

            # 新数据加入缓存
            buffer += chunk

            # 当缓存 >= 目标大小，就切分一次
            while len(buffer) >= target_size:
                # 找最后一个换行符
                split_pos = buffer.rfind(b"\n", 0, target_size)

                if split_pos == -1:
                    split_pos = target_size  # 实在没换行，强制截断

                # 切出当前文件
                current_chunk = buffer[:split_pos]
                buffer = buffer[split_pos:]  # 剩下的留在缓存，下一轮用

                # 写入
                output_path = f"bpe/data/owt_chunk_{file_index}.txt"
                with open(output_path, "wb") as f_out:
                    f_out.write(current_chunk)

                print(f"已生成：{output_path}  大小：{len(current_chunk) / 1024 / 1024:.2f}MB")
                file_index += 1

    print("\n✅ 全部切割完成！无死循环，所有文本完整！")

