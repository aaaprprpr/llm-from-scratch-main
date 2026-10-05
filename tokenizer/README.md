# 分词器

现有词表放在 `bpe_8192/` 与 `bpe_24576/`，运行时通过 `from tokenizer import Tokenizer` 加载。当前预训练、SFT、DPO 配置默认使用 24,576 词表；已有权重必须配套训练时的词表，不能仅修改路径来替换。

训练新词表的入口：

```bash
python -m tokenizer.sample_corpus
python -m tokenizer.train_bpe
```

采样与 BPE 参数集中在 [training_config.json](training_config.json)，相对路径以 `tokenizer/` 为基准。先配置采样输入或已有文本文件，再运行对应步骤；已有两套词表无需重新生成。特殊标记包括 `<|endoftext|>`、`<|im_start|>`、`<|im_end|>`。

新的分词器要在预训练开始前确定，并重新构建对应 token bin。当前配置面板编辑的是根目录 `configs/`，词表训练配置仍在本目录维护。
