# 本地语料清洗台

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

网页左侧选择“导入数据”，即可从空目录完成：

1. 点击“选择文件夹”选择本地 Hugging Face `save_to_disk` 数据集；JSONL/TXT 使用文件选择窗口。
2. 后端自动识别记录数、正文列和稳定 ID。只有识别错误时才需要展开高级修改。
3. 点击“导入数据集”，生成 Raw Snapshot 和一进一出的 Review Snapshot。
4. 新建或选择 Project，为全部文档创建人工清洗任务。

原生选择窗口由运行 FastAPI 的机器打开，不会把几十 GB 数据上传到浏览器。无桌面环境的 Ubuntu 可以展开“无法打开选择窗口”，手动填写服务器路径。

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
npm run build
```
