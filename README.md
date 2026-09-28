# llm-from-scratch

手写 Transformer 语言模型项目，包含预训练、SFT、DPO，以及 Hugging Face / vLLM 导出适配。模型实现见 `models/`；训练配置集中在 `configs/`，相对路径均以仓库根目录为基准。

这是学习项目。代码优先清楚地展示模型、数据和训练算法；只保留避免错误数据、错误训练结果或意外覆盖产物所需的检查。

## 目录

```text
models/                Transformer、注意力、RoPE、KV cache
tokenizer/             分词器加载与 BPE 词表训练
dataset/data_pipeline/ 数据下载、清洗与 token bin 构建
dataset/label/         语料标注与清洗台
train/pretrain/        预训练、推理、评估与纯权重导出
train/sft/             监督微调与数据适配
train/dpo/             DPO 训练与数据适配
export/hf/            Hugging Face 格式导出与加载
export/vllm/          vLLM 推理示例
configs/              各阶段配置
output/               本地日志、checkpoint 与评测产物（不纳入 Git）
```

`tokenizer/` 是 Python 包：`from tokenizer import Tokenizer` 加载运行时封装。两版词表分别放在 `bpe_8192/` 和 `bpe_24576/`；采样、训练入口及配置见 [tokenizer/README.md](tokenizer/README.md)。

## 从仓库根目录运行

先安装 `requirements.txt` 中的依赖，准备与配置匹配的数据和分词器。训练与导出入口：

```bash
python -m dataset.data_pipeline.download
python -m dataset.data_pipeline.build_bin
python -m train.pretrain.run_train_model --config configs/pretrain.json
python -m train.sft.download
python -m train.sft.sft
python -m train.dpo.dpo
python -m train.dpo.export
python -m export.hf.export_dpo
python -m export.hf.play_dpo_model
python -m export.vllm.serve_dpo
```

每个阶段的输入和输出路径见 `configs/*.json`。当前 `configs/sft.json` 的历史预训练 checkpoint 与数据集目录在本机尚不存在；其模型结构也与现有两版 MiniMind 权重不同，启动 SFT 前需要先确定兼容的基座配置。预训练说明见 [train/pretrain/README.md](train/pretrain/README.md)，数据入口见 [dataset/data_pipeline/README.md](dataset/data_pipeline/README.md)，导出模型的 vLLM 说明见 [export/vllm/README.md](export/vllm/README.md)。
