# 本地语料审核台

当前 V0.1 已跑通：

```text
Inspect / Preview
  → Import immutable Raw Snapshot
  → Prepare one-to-one Review Snapshot
  → Project + seeded Queue
  → Document / Block Review + Undo
  → deterministic Materialize
  → data_pipeline/build_bin.py
```

正文保存在 HF Dataset/Arrow，SQLite 只保存项目、抽样队列、标注当前状态和 append-only event；不会把千万条正文塞进数据库。

## 启动审核台

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

网页右上角选择“导入 / 建队列”，即可从空目录完成：

1. 选择本地 Hugging Face `save_to_disk`、JSONL 或 TXT。
2. 填写运行 FastAPI 的机器上的本地路径并检查字段。
3. 设置正文列、ID/标题/URL/metadata 映射，查看 3 条最终正文预览。
4. 导入 Raw Snapshot，并生成一进一出的 Review Snapshot。
5. 新建或选择 Project，挂载快照并创建固定种子随机队列。

已经成功导入的 Source/Prepare 会显示在“已有快照”中；刷新网页后可以直接继续，不会重复导入。

## CLI

查看所有命令：

```powershell
.\.venv\Scripts\python.exe -m label.cli --help
```

几十 GB 的 Import/Prepare 在网页中是同步长请求，页面会显示当前阶段，后端终端显示已处理文档数量和吞吐。不要在运行中关闭 FastAPI；若浏览器刷新，已经原子提交完成的快照仍会出现在“已有快照”中。

Mapping 是普通 JSON，例如：

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
→ queue-uniform/queue-quota → show-item/review-* → materialize
```

Materialize 输出仍保留 canonical provenance 列。`data_pipeline/build_bin.py` 只在检测到标注输出专用的 `dataset.json` 时启用新 loader；原有 `save_to_disk` 预处理结果继续走原来的 `datasets.load_from_disk`，并继续要求只有 `text` 一列。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s label\tests -v
cd label\frontend
npm run build
```
