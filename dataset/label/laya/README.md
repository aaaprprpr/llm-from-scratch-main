# Laya 语料清洗

本目录收集 Laya 的本地运行时、清洗任务微调、评测脚本和实验产物。清洗台的接入点仍在 [`../backend/laya_cleaning.py`](../backend/laya_cleaning.py)，模型来源 ID、分类问题和清洗算法沿用现有设置。Laya 在清洗进程内加载，无需单独启动模型服务。

## 目录

| 目录 | 用途 |
| --- | --- |
| `runtime/` | Laya 0.3.22 运行时，来源提交 `6d942c9`；授权见 [LICENSE](runtime/LICENSE) |
| `training/` | 训练数据构建、单设备微调、二分类评测与固定样本核看 |
| `experiments/` | 提问方式、整篇预览与三分类探索 |
| `models/` | 基础权重和清洗台当前使用的微调权重，本地准备，不纳入 Git |
| `data/` | 训练数据、候选 checkpoint、评测 JSON 与全文对照快照，不纳入 Git |

正式清洗数据仍保存在 `dataset/label/data/`：SQLite 审核与队列、Raw/Review Snapshot、批量进度、模型报告缓存都使用原来的位置。这里的 `data/` 只保存 Laya 训练和实验产物。

## 权重与实验快照

`models/laya_multilingual/` 是公开的 multilingual 基础权重。它是任务分类底座，零样本清洗表现见 [基础评测](../../../docs/历史记录/2026-09-30-Laya基础分类权重评测.md)。

`models/laya_wiki_cleaning_v1/` 是清洗台当前使用的二分类权重：以 4,000 个保留块和 4,000 个删除块微调两轮，每个换行块判断保留或删除。训练输入、完整留出评测及局限见 [微调评测](../../../docs/历史记录/2026-09-30-Laya语料清洗微调评测.md)。该分类器只能整块决策，无法执行块内局部编辑。

| `data/` 内的快照 | 关系与用途 |
| --- | --- |
| `laya_experiment_v1/`～`laya_experiment_v4/` | 共用同一份 8,000 块训练数据，保存不同阶段的试验和评测；v4 保留完整评测快照，不能理解成四版独立训练 |
| `laya_experiment_v5/` | 20,000 块二分类扩展试验，未采用为当前清洗权重 |
| `laya_ternary_v1/` | 保留、删除、局部编辑三分类试验，未接入当前清洗调度 |

快照目录中的历史名字保留，便于对照原评测、manifest 和权重哈希。目录位置变更不会重新训练模型或重写清洗结果。

## 脚本入口

所有命令从仓库根目录执行，通过 `python -m` 加载模块：

| 模块 | 用途 |
| --- | --- |
| `dataset.label.laya.training.dataset` | 从现有清洗报告与审核记录构建训练及留出数据 |
| `dataset.label.laya.training.finetune` | 单设备微调，保留 `--data`、`--model-dir`、`--output-dir` 等参数 |
| `dataset.label.laya.training.evaluate` | 二分类评测与阈值对照 |
| `dataset.label.laya.training.audit` | 重建固定核看样本；标签在 `training/audit_labels.json` |
| `dataset.label.laya.experiments.question_probe` | 基础权重与不同提问方式的对照 |
| `dataset.label.laya.experiments.document_preview` | 按实际清洗切分推理并组装全文对照 |
| `dataset.label.laya.experiments.ternary_dataset` | 构建三分类探索数据 |
| `dataset.label.laya.experiments.ternary_evaluate` | 评测三分类与局部编辑回退量 |

重建 8,000 块二分类试验的命令如下。这些脚本读取正式数据，产物写入新建的实验目录：

```bash
python -m dataset.label.laya.training.dataset --output dataset/label/laya/data/laya_rebuild --train-per-class 4000
OMP_NUM_THREADS=8 python -m dataset.label.laya.training.finetune --data dataset/label/laya/data/laya_rebuild/train_plain.jsonl --model-dir dataset/label/laya/models/laya_multilingual --output-dir dataset/label/laya/data/laya_rebuild/checkpoint_plain --epochs 2 --seed 23
python -m dataset.label.laya.training.evaluate --model-dir dataset/label/laya/data/laya_rebuild/checkpoint_plain --dataset dataset/label/laya/data/laya_rebuild/human_test.jsonl --output dataset/label/laya/data/laya_rebuild/eval_human_test.json
```

目录整理说明见 [离线 HTML](../../../docs/操作参考/Laya目录整理.html)，正式运行入口见 [清洗台操作参考](../../../docs/操作参考/语料清洗台.md)。
