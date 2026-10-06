# llm-from-scratch

手写 Transformer 语言模型学习项目，包含预训练、SFT、DPO、语料清洗和生态导出。代码按约定的数据与 checkpoint 格式直接运行，优先展示算法。

本文件是统一操作入口；算法笔记和历史结果见 [文档目录](docs/文档目录.md)。

## 运行与目录

使用项目的 Python 环境安装依赖，以下命令均从仓库根目录运行：

```bash
python -m pip install -r requirements.txt
```

`configs/*.json` 中的相对路径以仓库根目录为基准，由 [配置加载器](configs/config_loader.py)解析。`tokenizer/training_config.json` 的路径以 `tokenizer/` 为基准。数据、权重、凭据和运行缓存由各机器自行准备，不随 Git 同步。

| 目录 | 内容 |
| --- | --- |
| `models/` | Transformer、注意力、RoPE、KV cache |
| `configs/`、`config_panel/` | 配置文件、加载器和独立可视化面板 |
| `tokenizer/` | 分词器与 BPE 训练 |
| `dataset/data_pipeline/` | 独立下载与中间 Dataset 到 bin 的编码 |
| `dataset/storage.py` | 通用本地 Dataset/Arrow 读取 |
| `dataset/label/` | 语料清洗台及独立清洗实验 |
| `train/pretrain/`、`train/sft/`、`train/dpo/` | 各阶段训练、试玩和项目格式导出 |
| `export/hf/`、`export/vllm/` | 标准 HF 转换与推理 |
| `dataset/label/laya/` | Laya 分类运行时、清洗微调与实验 |
| `pipeline_audit/` | 从 bin 到 checkpoint 的链路审查 |
| `docs/` | 学习笔记、操作参考和历史记录 |
| `output/` | 训练日志、checkpoint 和评测产物 |

## 配置面板

```bash
python -m config_panel
```

打开 `http://127.0.0.1:8010`，选择模型结构、预训练、SFT、DPO、构建 bin 或数据下载配置。支持参数表单、搜索和原始 JSON；停止编辑约 500 毫秒后自动写回选中的配置文件，无需手动保存，下次启动任务时生效。端口可用 `--port` 修改。

前端在 `config_panel/frontend/`，通过独立 HTTP API 读写配置，无需构建。语料清洗有自己的设置页，不在这个训练面板里。

模型结构集中在 [configs/model.json](configs/model.json)，三阶段通过 `paths.model_config` 引用同一文件，层数、宽度、头数、上下文和词表大小只改一处。新预训练读取共用结构；SFT/DPO 接着读取该权重的 `model_args`，已有 checkpoint 的实际维度继续以权重记录为准。换基座时同步选择对应分词器和对话模板。

## 训练数据

正式数据路线是：来源导入 → 正文格式适配与 Prepare → 清洗台审核/清洗 → 中间 Dataset → 构建 bin。正文规范化在审核之前完成，bin 阶段只编码准备好的正文。

bin 使用独立的 [configs/build_bin.json](configs/build_bin.json)。`build_bin.input` 是本地 Dataset/Arrow 目录，里面包含字符串 `text` 列，每条记录是一篇完整正文；其他来源信息列可保留。通用读取器兼容 HF `save_to_disk` 目录和已有的 `dataset.json` + Arrow 分片目录，不依赖下载器或清洗台后台。

```bash
python -m dataset.data_pipeline.build_bin
```

默认中间目录为 `dataset/data_pipeline/data/intermediate/`，默认 bin 输出为 `dataset/data_pipeline/data/bins/`。配置中填写实际成品目录后再运行；训练时将 `configs/pretrain.json` 的数据路径指向本次输出。构建按整条记录划分、打乱、分词并追加 EOS，正文不截断、不补齐。`.meta.json` 记录 dtype、token 数、Dataset 指纹和分词器指纹。

旧规则 `preprocess.py` 已移除。原始 MiniMind JSONL、来源名、旧 `preprocess` 别名不再是 bin 输入；清洗台目前的 `effective_cleaned.jsonl` 也需在上游统一成中间 Dataset。本轮只收口 bin 端，上游下载与存储格式的统一另行整理。

原始下载仍由 [configs/data_pipeline.json](configs/data_pipeline.json) 的 `sources` 和 `download.sources` 控制，命令为 `python -m dataset.data_pipeline.download`。`dataset_samples/<source_id>.sample.json` 用于查看字段，不参与训练。

## 分词器

现成词表在 `tokenizer/bpe_8192/` 和 `tokenizer/bpe_24576/`，当前训练默认使用 24K。运行时可通过 `from tokenizer import Tokenizer` 加载。训练新词表时修改 [training_config.json](tokenizer/training_config.json)：

```bash
python -m tokenizer.sample_corpus
python -m tokenizer.train_bpe
```

换词表后重新构建 bin；已有权重继续使用训练时的词表。ChatML 特殊标记为 `<|im_start|>`、`<|im_end|>`，文档 EOS 为 `<|endoftext|>`。

## 预训练

模型结构在 [configs/model.json](configs/model.json)，优化器、学习率和训练预算在 [configs/pretrain.json](configs/pretrain.json)。

```bash
python -m train.pretrain.run_train_model --config configs/pretrain.json
python -m train.pretrain.play_model
python -m train.pretrain.export
```

`paths.resume=null` 新建 run，填写完整 checkpoint 路径则恢复训练；日志与权重写入 `paths.out_root`。入口默认按约一遍 token 文件计算预算，恢复最终 checkpoint 不会自动增加训练轮次。评估使用固定抽样位置，记录在 run 配置的 `runtime.evaluation_sampling`；改变 bin 或抽样设置后，旧 loss 曲线不能直接当同一评估集合比较。

预训练试玩和导出的 checkpoint 在各脚本顶部指定；试玩的 `CHECKPOINT_PATH=None` 才自动选择最新权重。预训练导出到 `output/pretrained_weights/model.pt`，是裸 state_dict；做 HF 转换时使用原完整 checkpoint。

## SFT

配置在 [configs/sft.json](configs/sft.json)，`paths.pretrained_weights` 选择预训练起点。默认下载 COIG-CQIA，子集由 `data.subsets` 选择；自备数据使用 `datasets.save_to_disk` 格式，字段为 `instruction`、`input`、`output`。

```bash
python -m train.sft.download
python -m train.sft.sft
python -m train.sft.play_sft_model
python -m train.sft.export
```

SFT 只对 assistant 回答和结束标记计算损失，超长答案跳过。`paths.resume` 恢复完整 SFT checkpoint；试玩和导出默认选择 `paths.sft_logs` 中最新 run 的最高步数权重。

## DPO

配置在 [configs/dpo.json](configs/dpo.json)。`paths.sft_checkpoint` 指定 SFT 起点，为 `null` 时从 `paths.sft_logs` 选最新权重。`paths.dataset` 支持 HF 数据集 ID 或本地 `save_to_disk` 目录，字段为 `prompt`、`chosen`、`rejected`，可选 `system`。

```bash
python -m train.dpo.dpo
python -m train.dpo.play_dpo_model
python -m train.dpo.export
```

DPO 使用可更新的 policy 和冻结的 SFT reference。`paths.resume` 恢复完整 DPO checkpoint，并沿用记录的 reference 路径，因此保留该 SFT 权重。试玩和导出优先读取 `paths.dpo_checkpoint`，未指定时从 `paths.dpo_logs` 选最新权重。

SFT/DPO 精简导出保留架构、配置等元信息，并附带分词器和模板，默认输出到 `output/sft_weights/`、`output/dpo_weights/`。继续训练使用日志里的完整 checkpoint。

## 生态导出

HF 默认输入和输出来自 `configs/dpo.json` 的 `paths.clean_weights`、`paths.hf_export`：

```bash
python -m export.hf.export_dpo
python -m export.hf.play_dpo_model
python -m export.vllm.serve_dpo
```

导出为标准 Llama 格式，包含 safetensors、分词器、聊天模板和训练元信息。精简 checkpoint 优先使用旁边的分词器和模板；直接输入训练 checkpoint 时使用 `configs/dpo.json` 的配置。指定 SFT 权重的例子：

```bash
python -m export.hf.export_dpo --checkpoint output/sft_weights/model.pt --output export/hf/sft_model
python -m export.hf.play_dpo_model --model export/hf/sft_model
python -m export.vllm.serve_dpo --model export/hf/sft_model
```

vLLM 在安装了相应依赖的算力环境运行；项目入口是离线生成示例。需要 HTTP 聊天接口时运行：

```bash
vllm serve export/hf/dpo_model --served-model-name scratch-chat
```

## 语料清洗台

清洗后端的批处理使用 Linux 文件锁，在 Linux 算力机的仓库根目录运行：

```bash
./clean
```

打开 `http://127.0.0.1:8000`，其他电脑用算力机 IP 访问。导入数据后生成简体 Review Snapshot，创建项目与队列，再进行人工编辑或自动清洗。默认数据目录是 `dataset/label/data`，可用 `LABEL_DATA_ROOT` 修改。

网页设置页选择单条模型、批量来源和付费失败补救开关；服务参数在 `configs/label.json`，凭据从根目录 `.env` 读取。批量任务可暂停续跑，人工编辑优先；合并语料 `effective_cleaned.jsonl` 是上游成品，进入 bin 前需统一成 Dataset/Arrow 目录。

CLI 查看入口和通用批量命令：

```bash
python -m dataset.label.cli --help
python -m dataset.label.cli llm-clean-batch --queue-id your_queue_id --limit 100
```

`--limit` 限制本次处理条数；`./clean-batch 100` 是当前 Wiki 队列的快捷入口。省略条数则继续处理整个队列。`cleaned.jsonl` 是批量中间结果，`materialize` 仅使用人工审核事件，两者与最终合并语料含义不同。导入、重置、续跑、产物和模型设置细节见 [语料清洗台操作参考](docs/操作参考/语料清洗台.md)。

修改清洗台前端时，在 `dataset/label/frontend/` 安装其 npm 依赖，运行 `npm run dev`；开发地址是 `http://127.0.0.1:5173`，`npm run build` 更新后端托管的 `dist/`。

## 网页聊天脚本

`deepseek-web-api/` 已删除。清洗台原有网页来源仍引用这些脚本，目前无法实际调用；正式 API、本地 Qwen 和 Laya 不依赖该目录。协议观察仅作为 [历史记录](docs/历史记录/2026-09-28-网页聊天接口请求记录.md)保留。

## 实验与链路审查

`pipeline_audit` 检查 bin、batch、next-token 标签、模型前向与 checkpoint，配置在 `pipeline_audit/config.json`：

```bash
python pipeline_audit/run_audit.py
```

报告写入 `pipeline_audit/reports/run_<时间>/`，包含汇总 JSON，以及相应输入可用时的 bin 样本 HTML、checkpoint 指标 CSV。缺少输入的项目会跳过；它不评价正文清洗质量。

`dataset/label/experiments/` 保存独立清洗探针：`eval_prose_cleaning.py` / `prose_extraction_v4/` 是人工对照，`history_cleaning_debug.py` / `history-cleaning-debug/` 是单篇调用记录；`fragment_editing_v3/`、`llm_cleaning_integration/` 和 `llamacpp_probe_20260913_013551/` 保存片段编辑、API 联调与本地模型探针。它们不参与训练入口，服务、队列和样本路径描述的是当时环境。

Laya 集中在 [dataset/label/laya/](dataset/label/laya/README.md)：`runtime/` 保存 Laya 0.3.22 运行时，来源提交 `6d942c9`，许可证见 [runtime/LICENSE](dataset/label/laya/runtime/LICENSE)；`training/` 保存数据构建、微调与评测脚本，`experiments/` 保存提问与三分类探针。基础权重在 `models/laya_multilingual/`，清洗台当前使用的 8,000 块二分类权重在 `models/laya_wiki_cleaning_v1/`；权重和 `data/` 下的实验快照均不纳入 Git。v1～v4 是同一份 8,000 块训练数据的不同实验阶段，v4 保留完整评测，不能当成四版独立训练。目录用途见 [Laya 目录说明](docs/操作参考/Laya目录整理.html)，任务效果与局限见文档目录里的历史评测。
