# 数据产物目录

这里存放原始下载、预处理结果、bin 和元数据；大文件由 Git 忽略。当前路径兼容已生成的
MiniMind full/mini 文件和已有训练配置，不在整理代码时搬动数据。

```text
data/
  downloads/minimind/      pretrain_t2t.jsonl（full）及 pretrain_t2t_mini.jsonl（mini）
  minimind_full/           当前 full 版 train.bin、val.bin、*.meta.json
  minimind/                历史 mini 版 train.bin、val.bin、*.meta.json
  .cache/                  下载过程的可再生成缓存
  preprocessed/            旧通用清洗流程的 Arrow 中间结果（当前未启用）
  train.bin、val.bin        旧通用流程的 bin（当前未启用）
```

当前预训练输入以 [`configs/pretrain.json`](../../configs/pretrain.json) 为准，指向
`minimind_full/`；构建参数在 [`configs/data_pipeline.json`](../../configs/data_pipeline.json)
的 `minimind_bin`。本机可能只保存 mini 原始文件和 mini bin；不要因为 full 路径在本机
不存在，就用 mini 文件覆盖 full 输出。完整用法见 [MiniMind 入口](../minimind/README.md)。

训练代码按 `uint16` 读取 bin，词表大小必须小于等于 65,536，且对应元数据中的
tokenizer SHA256 必须与 `bpe/tokenizer_24576` 一致。
