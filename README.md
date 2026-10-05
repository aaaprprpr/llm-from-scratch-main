# llm-from-scratch

手写 Transformer 语言模型项目，包含预训练、SFT、DPO，以及 Hugging Face / vLLM 导出适配。模型实现见 `models/`；训练配置集中在 `configs/`，相对路径均以仓库根目录为基准。

这是学习项目。代码优先清楚地展示模型、数据和训练算法，直接按各模块约定的数据与 checkpoint 格式运行。

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
config_panel/         独立的可视化配置面板
docs/                 中文学习笔记与历史实验记录
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
python -m train.sft.export
python -m train.dpo.dpo
python -m train.dpo.export
python -m export.hf.export_dpo
python -m export.hf.play_dpo_model
python -m export.vllm.serve_dpo
```

每个阶段的输入和输出路径见 `configs/*.json`。运行 `python -m config_panel` 后，打开 `http://127.0.0.1:8010` 可视化编辑配置；使用方式见 [配置面板](config_panel/README.md)。

SFT 从预训练 checkpoint、DPO 从 SFT checkpoint 读取模型结构，不再维护重复的网络参数。当前 SFT 默认指向 24K MiniMind 完整版基座，并使用对应分词器。本机尚未附带该 checkpoint 和 SFT 数据；训练前准备配置里的输入路径。DPO 初次运行默认查找最新 SFT 权重，恢复时沿用原固定 reference。

操作说明：[预训练](train/pretrain/README.md)、[数据流水线](dataset/data_pipeline/README.md)、[SFT](train/sft/README.md)、[DPO](train/dpo/README.md)、[Hugging Face 导出](export/hf/README.md)、[vLLM](export/vllm/README.md)。算法笔记与历史实验见 [文档目录](docs/文档目录.md)。
