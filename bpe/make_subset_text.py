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

import json,os
if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    size_subset = 128
    PATH_OWT = "C:/Users/guojia/PycharmProjects/heima/llm-from-scratch-main/wiki_cn_clean.txt"
    PATH_OWT_SUBSET = f"data/owt_subset_{size_subset}mb.txt"

    with open(PATH_OWT, "rb") as f_in:
        chunk = f_in.read(size_subset * 1024 * 1024) # only read 1024 mb
        with open(PATH_OWT_SUBSET, "wb") as f_out:
            f_out.write(chunk)


