# 数据集结构样本

`data_pipeline/download.py` 下载或发现本地已存在的数据集后，会在这里生成：

```text
<source_id>.sample.json
```

这个文件只用于快速查看数据集结构，通常包含：

- 数据源 id；
- dataset/config/split；
- adapter；
- split 行数和字段；
- 前几条样本。

它不是训练输入。真正的训练输入由 `data_pipeline/preprocess.py` 从 `data_pipeline/data/downloads/...` 读取并清洗生成。
