# 本地语料清洗台

历史 LLM 清洗探针与人工对照记录见 [experiments](experiments/README.md)。

当前 V0.1 已跑通：

```text
Inspect / Preview
  → Import immutable Raw Snapshot
  → Prepare one-to-one Review Snapshot
  → Project + seeded Queue
  → Full Dataset Cleaning + Direct Text Editing + Undo
  → deterministic Materialize
  → data_pipeline/build_bin.py
```

正文保存在 HF Dataset/Arrow，SQLite 只保存项目、全量清洗任务、实际标注状态和 append-only event；不会把千万条正文或待办索引塞进数据库。

## 启动清洗台

所有 Python 命令都显式使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m uvicorn label.backend.api:create_app --factory --host 127.0.0.1 --port 8000
```

Ubuntu 对应命令：

```bash
./.venv/bin/python -m uvicorn label.backend.api:create_app --factory --host 127.0.0.1 --port 8000
```

前端开发模式（另开终端）：

```powershell
cd label\frontend
npm run dev
```

打开 `http://127.0.0.1:5173`。也可以先运行 `npm run build`，此时 FastAPI 会直接托管 `dist/`，打开 `http://127.0.0.1:8000` 即可。

默认数据根目录是 `label/data`，可通过 `LABEL_DATA_ROOT` 修改。

## LLM API 清洗本条

清洗工作区支持直接改字、选区删除、拖动选中文字到另一段中拼接、整段拖动重排、繁简转换、撤销和重做。最终保存的是编辑拼接后的正文。

打开每条文档时，使用固定版本 OpenCC 1.3.0 的 `t2s` 配置自动转简体，已保存的人工正文和可恢复的删除块也采用同一转换。转换在加载时完成，不在每次输入或撤销时重复执行，也不进入撤销历史。`t2s` 做繁简转换，不额外将“软体／滑鼠”等词替换成地区术语；粘贴繁体内容后仍可手动点击“繁体转简体”。保存时写入当前简体草稿，加载本身不写人工审核记录。

右侧默认显示 **简体原文与当前草稿** 的逐字删改对照，红色表示删除/移出，绿色表示插入/移入。改字、拼接、重排、LLM 编辑和撤销都会更新；也可切换“简体原文”“拼接结果”。差异以原始快照转换后的简体文本为基准，因此包括已保存的历史编辑，但不把原始繁体到简体的转换算作删改。原始快照及来源校验哈希保持不变。大范围改动退回行级对照，计算在 Web Worker 中执行。

拖动正文与右侧栏之间的分隔条调整宽度，宽度自动记住；双击恢复默认，也可聚焦分隔条后使用左右方向键。窄屏自动改为上下排列。

正文上方点击 **LLM 清洗本条**，处理当前尚未保存的草稿，已手动删除的段落不会重新送入模型。LLM 返回可继续人工加工的编辑草稿及理由。

- 当前清洗版本 `prose_extraction_v5` 以已完成的《数学》《哲学》人工结果为标准：保留具体论述，删除独立标题、目录名单、参考书目、外链和残破 Wiki 表格；有解释的列举仍保留。《文学》未清完，不用作标准。
- 模型用 `removals` 删除整片，用 `edits` 裁剪片内文字和调整空白，用 `joins` 连接明确需要拼接的相邻保留片段。未列出的片段自动按原顺序保留。后端拒绝新增文字、改写、重排、重复或跳过保留片段的拼接方案。长文逐块处理，跨分块提供附属栏目/表格位置提示；拼接限于块内。
- 判为“待定”的分块原样保留。提示词和校验不能保证语义判断正确，漏删标题和误删正文仍需人工对照修正。
- 结果只进入可撤销的编辑草稿。点击“撤销本次清洗”或按 Ctrl Z 恢复；继续人工编辑后仍可逐步撤销。人工质量评分、类别不会被覆盖。
- 审核后按 ↑ 保留并保存，Ctrl D 丢弃整条。沿用原有翻页行为：→ 会保存为当前决定，未审核条目保存为待定。刷新页面会丢失尚未保存的草稿。
- JSON 格式、引用编号、局部编辑或输出长度校验失败时，带上错误原因仅重试当前分块一次，已成功的分块不会重跑。重试仍失败时显示具体分块位置，整次建议不应用，原草稿保留。连接失败不自动重试。

服务配置在 `configs/label.json`，默认使用 DashScope 的 `qwen-plus`。后端自动读取**项目根目录**的 `.env`（不受启动目录影响），系统环境变量优先：

```dotenv
DASHSCOPE_API_KEY=你的百炼API密钥
DEFAULT_MODEL=qwen-plus
# 可选：按密钥所属地域覆盖 API 地址
# DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

`.env` 已加入 Git 忽略；可参考根目录 `.env.example`。依赖中新增 `python-dotenv`，已有环境执行 `python -m pip install python-dotenv`。修改配置或 `.env` 后重启后端。

远程模式只调用 `/chat/completions`，使用 Bearer 认证、关闭思考和 JSON 输出模式；Schema 放在提示词中，返回后仍执行原有 Pydantic、片段编号及只删不改写校验。参数格式依据 [DashScope Chat API 文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)。密钥仅由后端读取，不进入前端或建议报告。鉴权、限流、额度及连接错误会显示在原有工作区中。

如需切回原有 4096 上下文的本地 llama.cpp，将 `provider` 设为 `llamacpp`、`base_url` 设为 `http://127.0.0.1:8080`、`model` 设为本地模型名，并恢复 `context_tokens=4096`、`max_output_tokens=1536`。该模式保留 `/props`、`/apply-template`、`/tokenize` 和 `/v1/chat/completions` 适配，且只允许回环地址，不读取 DashScope 环境配置。

长文按句子和行拆分，成对中文引号内的问号、句号和换行不作为普通切分点，避免将同一引文拆散后引发片段编号错配。超长引文仍受单片长度和上下文上限约束。逐块串行处理全部正文，不静默截断。远程模式按包含 Schema 的完整提示词 UTF-8 字节数保守估算输入预算，预留输出及重试空间，不依赖本地 tokenizer；实际 token 用量以 API 返回值为准。本地模式仍使用服务的聊天模板与 tokenizer 计算。当前远程配置为 32768 上下文预算、4096 输出 token、10 万字符、64 个分块，单次请求超时 120 秒；长文可能花几分钟。超限会明确报错。

每次完整成功的建议单独保存到 `label/data/llm_suggestions/<suggestion_id>.json`（使用自定义数据根目录时随根目录移动），记录原始草稿及分隔符、来源、模型、提示词版本、删除位置、局部裁剪、拼接计划、重试原因及结果。它不会自动写入人工审核表或成为已接受的训练数据。后续可将这些建议与人工最终结果对照，积累快速分类器的训练样本。

本次工作区只导入 `data_pipeline/data/downloads/wiki_zh_20231101`，项目为 `wiki_zh_20231101`，全量人工队列为 `wiki_zh_20231101_full`。正文映射 `text`，`title`、`url`、`id` 保存为来源信息。队列包含全部 1,384,748 条文档；点击 **LLM 清洗本条** 才会调用 API 处理当前文档，创建队列不会自动批量调用模型。

网页左侧选择“导入数据”，即可从空目录完成：

1. 点击“选择文件夹”选择本地 Hugging Face `save_to_disk` 数据集；JSONL/TXT 使用文件选择窗口。
2. 自动识别记录数、内容格式、正文列和稳定 ID，显示前三条正文预览。内容格式与字段映射可直接调整，调整后重新预览。
3. 点击“导入数据集”，生成 Raw Snapshot 和一进一出的 Review Snapshot。
4. 新建或选择 Project，为全部文档创建人工清洗任务。

原生选择窗口由运行 FastAPI 的机器打开，不会把几十 GB 数据上传到浏览器。无桌面环境的 Ubuntu 可以展开“无法打开选择窗口”，手动填写服务器路径。

内容适配复用 `data_pipeline/record_adapters.py`：文章、分类正文、instruction/input/output、问答、问答含 think/reasoning、ShareGPT、role/content 对话（含 messages）、贴吧主楼与回复。只抽取并拼接原有文字，不生成 SFT/DPO 格式；role、分类标签不会混入正文。自定义映射仍支持点路径字符串列；直接选中数组或对象会报错，避免只留下标题而静默丢失正文。

API/CLI 的 mapping 可指定 `record_adapter` 和空 `text_fields`；原有纯字段映射不需修改，旧快照的版本标识保持兼容。这里只做内容格式适配，Import 不执行旧预处理的过滤规则。

已经成功导入的 Source/Prepare 会显示在“已有快照”中；刷新网页后可以直接继续，不会重复导入。

## CLI

查看所有命令：

```powershell
.\.venv\Scripts\python.exe -m label.cli --help
```

几十 GB 的 Import/Prepare 在网页中是同步长请求，页面会显示当前阶段，后端终端显示已处理文档数量和吞吐。不要在运行中关闭 FastAPI；若浏览器刷新，已经原子提交完成的快照仍会出现在“已有快照”中。全量清洗任务使用虚拟 ordinal 映射，不会把上千万条待办索引预先写入 SQLite；SQLite 只记录实际发生的人工操作。

CLI 中的 Mapping 是普通 JSON，例如：

```json
{
  "text_fields": ["title", "content"],
  "title_field": "title",
  "local_id_field": "id",
  "metadata_fields": ["dataType"]
}
```

完整的核心命令顺序：

```text
inspect → preview → import → prepare → project-create → source-attach
→ queue-full → show-item/review-* → materialize
```

Materialize 输出仍保留 canonical provenance 列。`data_pipeline/build_bin.py` 只在检测到标注输出专用的 `dataset.json` 时启用新 loader；原有 `save_to_disk` 预处理结果继续走原来的 `datasets.load_from_disk`，并继续要求只有 `text` 一列。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s label\tests -v
cd label\frontend
npm test
npm run build
```
