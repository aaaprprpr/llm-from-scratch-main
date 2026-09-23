# 标注清洗实验记录

这里保存标注工具和 LLM 清洗器的历史探针、人工对照与结果，**不参与预训练或
`pipeline_audit/run_audit.py`**。正式标注数据仍在 `label/data`，这里的 JSON 是实验记录。

| 内容 | 用途 |
| --- | --- |
| `eval_prose_cleaning.py`、`prose_extraction_v4/` | 将清洗建议与前两条人工审核正文对照；`comparison.md` 说明局限。 |
| `history_cleaning_debug.py`、`history-cleaning-debug/` | 单条历史正文的清洗调用记录。 |
| `fragment_editing_v3/` | 片段编辑示例与建议输出。 |
| `llm_cleaning_integration/` | 本地标注 API 的清洗联调记录。 |
| `llamacpp_probe_20260913_013551/` | 本地 llama.cpp 质量探针结果。 |

这些脚本包含当时的本地服务地址、队列 ID 和样本位置，运行时可能调用 LLM API 并写入
本目录的结果文件。它们是可追溯的历史实验，不是当前训练入口；先检查本机服务和数据
是否仍匹配，再手动运行。
