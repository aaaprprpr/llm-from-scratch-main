# 数据产物目录

这里存放下载原文、预处理结果、bin 和元数据；大文件由 Git 忽略。原有训练数据现位于本目录。

```text
data/
  downloads/minimind/       MiniMind full/mini 原始 JSONL
  minimind_full_8192/       当前 8K 词表的 full train.bin、val.bin 和元数据
  minimind_full/            历史 24K 词表的 full bin
  minimind/                 历史 mini bin
  preprocessed/             可选 Hugging Face 数据集清洗结果
  .cache/                   可再生成的下载缓存
```

当前训练输入见 [`configs/pretrain.json`](../../../configs/pretrain.json)；构建参数见 [`configs/data_pipeline.json`](../../../configs/data_pipeline.json) 的 `build_bin`。bin 的 dtype 和分词器 SHA256 写在各自的 `.meta.json`；训练时必须使用与 bin 匹配的 tokenizer。
