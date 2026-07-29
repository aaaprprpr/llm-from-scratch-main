# 数据目录

该目录保存数据流水线的输入和产物：

- `raw/*.txt`：UTF-8 原始文本；
- `train.txt`、`val.txt`：清洗和划分后的文本；
- `train.bin`、`val.bin`：供预训练读取的平铺 token id。

下载、预处理和生成 bin 的统一命令见 [`data_pipeline/README.md`](../README.md)。

当前训练代码使用 `np.uint16` 读取 bin，因此词表必须小于 65,536，且生成 bin 时应确认元数据中的 dtype 为 `uint16`。
