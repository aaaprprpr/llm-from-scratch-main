# LLM 数据整理与人工标注平台设计

## 0. 结论与范围

目标是在现有预训练管线前增加一个可追溯的数据资产层，并提供高效率的人工审核工具。

平台首先解决以下问题：

- 将 Hugging Face Dataset、TXT、JSONL 等来源导入统一格式。
- 在几十 GB、上千万文档规模下随机访问和抽样，不把全文加载进内存。
- 让人看到经过自动归一化后、真正准备交给 tokenizer 的文本。
- 快速标记 Document 的 Keep / Drop / Unsure、质量和原因。
- 删除低质量 Block；后续再增加 Span 编辑。
- 原始数据只读，人工行为可追踪、可撤销、可重复物化。
- 输出 Hugging Face Dataset，继续交给现有 `build_bin.py`。
- 为未来规则模型、LightGBM 或小型 Encoder 积累可靠监督数据。

V0.1 不以“实现通用数据编辑器”为目标。面对上千万条记录，人不可能逐条精修；第一优先级是抽样、快速判定、来源对比和形成可信标签。

---

## 1. 管线边界

### 1.1 当前管线

```text
download.py
    ↓
Downloaded HF Dataset
    ↓
preprocess.py
    ↓
Normalized / filtered Dataset
    ↓
build_bin.py
    ↓
train.bin / val.bin
```

`build_bin.py` 当前已经承担清晰且单一的职责：文档级 train/val 划分、tokenize、追加 EOS、写连续 token binary。标注平台不处理 bin，也不修改这部分。

### 1.2 新管线

```text
External Sources
    ↓
Streaming Source Adapter
    ↓
Canonical Raw Snapshot（只读、保留来源）
    ↓
Deterministic Prepare（schema adapter + preprocess 规则）
    ↓
Review Snapshot（人工看到的文本）
    ↓
Sampling Queue + Human Review
    ↓
Materialize at event_seq N
    ↓
Curated HF Dataset
    ↓
Validation-only check
    ↓
build_bin.py
```

关键原则：人工审核必须发生在会改变正文的自动归一化之后。否则 UI 中看到的是 A，`preprocess.py` 又通过 ftfy、Unicode normalize 等操作把它变成 B，最终无法确认模型真正吃到了什么。

因此：

- Raw Snapshot 保存导入结果和 provenance。
- Review Snapshot 保存自动归一化后的正文，是人工编辑的不可变基底。
- Materialize 后不得再静默修改正文，只允许验证和拒绝非法记录。
- `build_bin.py` 只读取 `text`，所以 Curated Dataset 可以继续保留 provenance 列。

现有 `preprocess.py` 暂时保留。实现阶段只需让它可选地保留 `doc_id/source_id/source_revision/source_row`，或提取其纯函数供 Prepare 阶段调用，不要在第一步整体重写。

---

## 2. 存储模型

采用：

```text
HF Dataset / Arrow：不可变正文快照
SQLite：项目、队列、当前标注状态、编辑事件、物化历史
```

Arrow 适合大文本、分片、memory mapping 和按 row 随机读取；SQLite 适合频繁的小事务。不要把几十 GB 正文复制进 SQLite，也不要为一次标注重写 Arrow。

推荐物理目录：

```text
label/
├── data/
│   ├── sources/
│   │   └── <source_id>/<source_revision>/
│   │       ├── manifest.json
│   │       ├── raw/
│   │       └── review/
│   ├── curation.sqlite3
│   └── exports/
│       └── <materialization_id>/
│           ├── dataset/
│           └── manifest.json
├── backend/
├── frontend/
└── tests/
```

不要用一个可变的 `canonical/current` 目录覆盖旧数据。每次导入形成不可变 `source_revision`；Project 固定引用具体 revision。

对于已经由 Hugging Face `save_to_disk()` 保存的大型 Dataset，Raw Snapshot 可以使用只读引用加 manifest，而不是机械复制几十 GB；只有无法保证外部路径不可变时才复制进受管目录。Review Snapshot 仍是独立、不可变的派生版本。

---

## 3. Canonical Dataset Schema

第一版固定 schema，避免每个数据源生成不同 Arrow struct：

```text
schema_version: string
doc_id: string
source_id: string
source_revision: string
source_row: int64
source_local_id: string | null
text: string
title: string | null
url: string | null
content_sha256: string
metadata_json: string
```

### 3.1 `doc_id`

`doc_id` 必须稳定、全局唯一，但不能只使用正文 hash。重复正文也可能来自不同来源，必须保留各自 provenance。

推荐生成方式：

```text
doc_id = blake2b_128(
    source_id + "\0" +
    source_revision + "\0" +
    stable_source_locator
)
```

`stable_source_locator` 优先级：

1. 来源自带且承诺稳定的主键。
2. JSONL 行号、TXT byte offset、HF split + row index 等不可变位置。
3. 最后才使用正文 hash + occurrence index。

正文内容另存 `content_sha256`。打开项目时必须校验 `doc_id` 对应的 `content_sha256`，不允许把旧标注悄悄套到新正文。

每一级 snapshot 的 `content_sha256` 都针对该级实际 `text`：Raw hash 和 Review hash 分别记录，人工 operation 只引用 Review hash。

### 3.2 `metadata_json`

异构 metadata 使用规范化 JSON 字符串，不使用随来源变化的 Arrow struct。常用筛选字段应提升为固定列；冷门字段留在 JSON 中。

### 3.3 Source Manifest

每个 source revision 必须包含：

```text
source_id
source_revision
source_type
original_location
downloaded/imported_at
license / terms（允许为空，但必须显式记录 unknown）
input fingerprint
adapter name + version
mapping config
prepare config hash
record count
content hash summary
```

---

## 4. Source Adapter

Adapter 不应该自行构建并返回完整 Dataset。它只负责检查输入和流式产生统一 Document，由中央 Import Service 分批写 Arrow、生成 ID 和 manifest。

概念接口：

```python
@dataclass(frozen=True)
class ImportedDocument:
    source_local_id: str | None
    text: str
    title: str | None = None
    url: str | None = None
    metadata: dict[str, object] | None = None


class SourceAdapter(ABC):
    adapter_name: str
    adapter_version: str

    def inspect(self, spec: SourceSpec) -> Inspection:
        ...

    def preview(
        self,
        spec: SourceSpec,
        mapping: FieldMapping,
        limit: int,
    ) -> list[ImportedDocument]:
        ...

    def iter_documents(
        self,
        spec: SourceSpec,
        mapping: FieldMapping,
    ) -> Iterator[ImportedDocument]:
        ...
```

Import Service 负责：

- schema 校验；
- `doc_id/content_sha256`；
- 分批 Arrow 写入；
- 原子提交临时目录；
- manifest；
- 失败恢复和进度记录。

### 4.1 第一版格式

- Hugging Face：本地 `save_to_disk` Dataset 和已下载的数据源。
- JSONL：逐行解析，是外部结构化交换首选。
- TXT：支持 `whole_file`、`one_line_per_document`、`blank_line_separator` 和自定义分隔符。
- JSON Array：小文件可直接解析；大文件必须使用 `ijson` 等流式解析器，禁止假装流式后调用 `json.load()`。

未知 schema 必须执行：

```text
Inspect → Field Mapping → Preview → Confirm → Import
```

Mapping Template 必须带 adapter version 和输入 schema fingerprint；字段变化时不能静默复用旧模板。

Web DOM 和 Browser Extension 放到 V0.2，不做通用爬虫。

---

## 5. Review Dataset 与 Block

### 5.1 Document

Review Dataset 一行就是一个训练 Document，最终在 bin 中由一个 EOS 与其他文档分隔。

Document 状态：

```text
unreviewed
keep
drop
unsure
```

质量分建议使用语义明确的四级，而不是只有数字：

```text
0 unusable     乱码、垃圾、无法形成训练文本
1 weak         基本可读，但模板化、营销化、重复或信息价值低
2 acceptable   连贯、正常、可作为普通预训练文本
3 high_quality 结构清晰、信息密度高、语言质量好
```

`decision`、`quality`、内容类型和缺陷必须分开，不能把所有信息压进一个 Keep/Drop 标签。

第一版增加 `primary_category`：

```text
encyclopedia
news
marketing
fiction
forum_or_social
qa_or_instruction
academic_or_technical
code
reference_or_table
other
```

再使用多选 `flags` 记录缺陷；Keep 文档也可以带 flags：

```text
advertisement
boilerplate
spam
garbled
duplicate
low_information
bad_format
unsafe_or_pii
wrong_language
irrelevant
other
```

分类词表和质量标准必须带 `guideline_version`。在正式大量标注前先制作一小批带正反例的校准集，避免同一个人隔几周后标准漂移。

### 5.2 Block parser

Block 是 Review Text 的确定性视图，不按固定字符数切分。

第一版规则：

1. Prepare 阶段已经统一换行为 `\n`。
2. 保留原始分隔符。
3. 普通文本优先按空行分段；连续非空短行可以按配置合并。
4. Web 来源未来可以直接携带 DOM block boundaries。
5. parser 必须有 `parser_version` 和单元测试。

Block ID：

```text
block_id = blake2b_128(
    doc_id + parser_version + start_cp + end_cp + block_content_hash
)
```

如果 Review Text 或 parser version 改变，生成新 Block ID，旧 operation 显式失效，而不是错贴到别的文字上。

### 5.3 Span offset

浏览器 JavaScript 的原生 offset 是 UTF-16 code unit，Python 切片使用 Unicode code point，含 emoji 时两者不同。API 不能含糊地使用 `start/end`。

统一规定：

```text
start_cp / end_cp = normalized block text 的 Unicode code point offset
```

前端使用 `Array.from(text)` 转换选区；请求同时携带：

```text
block_id
base_block_hash
start_cp
end_cp
selected_text_sha256
```

后端再次验证边界和 hash。

### 5.4 V0.1 编辑能力

第一版只支持：

- Document Keep / Drop / Unsure；
- quality、primary category、flags、notes；
- Block Keep / Drop；
- 撤销最近操作。

Span delete/replace、Block move/merge/split 不进入 V0.1。它们会显著增加 offset、冲突、撤销和 materialize 复杂度，而且对千万级预训练语料的单位时间收益很低。

V0.2 增加 Span edit 时，所有 Span edit 都引用不可变原始 Block；同一 Block 的区间不允许重叠，materialize 时从右向左应用。Block Drop 优先于 Block 内 Span edit。

---

## 6. SQLite 模型

SQLite 使用 WAL、foreign keys 和显式 schema migration。正文不进入 SQLite。

核心表：

```text
projects
    project_id PK
    name
    created_at
    current_revision

project_sources
    project_id
    source_id
    source_revision
    dataset_path
    row_count

review_queues
    queue_id PK
    project_id
    sampling_policy_json
    sampling_seed
    created_at

queue_items
    queue_id
    ordinal
    source_id
    source_revision
    source_row
    doc_id
    priority
    state

document_reviews
    project_id
    doc_id
    source_row
    content_sha256
    decision
    quality
    primary_category
    flags_json
    notes
    guideline_version
    revision
    updated_at

block_reviews
    project_id
    doc_id
    block_id
    base_block_hash
    decision
    reason
    revision

events
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT
    project_id
    entity_type
    entity_id
    operation_type
    payload_json
    actor
    created_at
    undo_of_event_seq

materializations
    materialization_id PK
    project_id
    snapshot_event_seq
    policy_json
    config_hash
    output_path
    manifest_sha256
    created_at
```

不要每次请求都重放完整 Event Log。`document_reviews/block_reviews` 保存当前状态用于快速读取；`events` 是 append-only 审计记录。一次操作在同一个事务里更新 current state 并追加 event。

若要重新生成当前状态，直接读取 current state 表；若要物化历史 `event_seq`，则在批处理开始时把不晚于该序号的 events 重放到临时状态，或读取已保存的状态快照后重放增量事件。不能用“最新状态 + 一个旧 event_seq”伪装成历史版本。

撤销通过追加补偿事件实现，不删除历史。API 修改必须携带当前 `revision`，不匹配时返回冲突，避免两个标签页互相覆盖。

不必把 1800 万文档全部复制成 SQLite index。Arrow row 本身支持 O(1) 读取；SQLite 只存进入抽样队列或已经操作的文档。定位键使用 `(source_id, source_revision, source_row, doc_id)`。

---

## 7. Sampling Queue

人工标注不能从“第一条一直翻到最后一条”。平台必须先创建可复现的 Queue：

```text
uniform_random
source_quota
length_stratified
rule_flagged
model_uncertainty（后期）
```

每个 Queue 记录 seed、来源配额、过滤条件和生成时间。UI 只浏览 Queue，不对 1800 万行做巨大的 OFFSET 分页。

第一批监督数据必须包含固定种子的均匀随机样本和独立 holdout。不能一开始只标模型认为“可疑”的文本，否则训练出的自动筛选器会有严重选择偏差。

推荐工作流：

```text
来源分层随机样本
    ↓
人工标注
    ↓
规则/模型候选
    ↓
人工复核
    ↓
持续保留随机审计样本
```

---

## 8. API 与 UI

推荐技术栈：

```text
FastAPI
React + TypeScript + Vite
HF datasets / PyArrow
SQLite
```

React 在文本选区、快捷键和局部状态方面有价值；FastAPI 负责唯一的数据访问和事务边界。前端不直接打开 SQLite 或 Arrow。

第一版主要 API：

```text
GET  /api/projects
POST /api/projects
POST /api/imports/inspect
POST /api/imports/preview
POST /api/imports
POST /api/queues
GET  /api/queues/{queue_id}/items/{ordinal}
PUT  /api/reviews/documents/{doc_id}
PUT  /api/reviews/blocks/{block_id}
POST /api/events/{event_seq}/undo
POST /api/materializations
GET  /api/materializations/{id}
```

Document 响应包含 rendered blocks、source 信息、当前 review 和 revision。

UI 采用单文档审核台，而不是先实现复杂文件管理器：

```text
┌──────────────┬──────────────────────────────┬──────────────┐
│ Queue/进度   │ Document + Blocks            │ Provenance   │
│ 来源配额     │                              │ URL/License  │
│ 快速筛选     │ Keep / Drop / Unsure         │ Review/Event │
└──────────────┴──────────────────────────────┴──────────────┘
```

快捷键：

```text
A       Keep
D       Drop
S       Unsure
0..3    Quality
X       Drop 当前 Block
Ctrl+Z  Undo
J / →   Next
K / ←   Previous
```

每次操作立即事务保存，UI 显示“已持久化”，不依赖离开页面时批量保存。

必须提供：

- Raw Text / Review Text 切换，默认展示 Review Text；
- Review Text 与最终 Materialized Text 预览；
- source/revision/row/url/license；
- 字符数和 tokenizer token 数；
- Block 删除前后 diff；
- 当前队列进度和来源分布。
- decision、quality、primary category、flags 的快捷标注。

V0.1 不需要一次渲染几千条虚拟列表。一个 Queue item 一次加载一个 Document，上一条/下一条即可；列表和搜索后续添加。

---

## 9. Materialize

Materialize 必须是确定性、可重放、原子提交的批处理。

输入：

```text
固定 source revisions
project_id
snapshot_event_seq
materialization policy
block parser version
代码版本/config hash
```

决策策略必须显式选择：

```text
keep_only
    只输出 decision=keep 的文档，适合构建小型高质量集。

drop_rejected
    输出所有未 drop 的文档，适合对大语料应用人工黑名单。
```

禁止把“未标注是否保留”写成隐式逻辑。

每个 Document 的物化顺序：

1. 读取并校验 `doc_id/content_sha256`。
2. 获取不晚于 `snapshot_event_seq` 的 review 状态。
3. 按 policy 决定是否输出。
4. 用固定 parser version 重建 Blocks。
5. 应用 Block Drop。
6. V0.2 起校验并应用非重叠 Span edits。
7. 使用固定分隔符重组正文。
8. 拒绝空正文和非法 Unicode。
9. 写临时 Arrow shards。
10. 生成 manifest 后原子 rename。

Materialization manifest 至少记录：

- 输入 source revisions；
- snapshot event seq；
- policy 和 parser version；
- 输入、keep、drop、unsure、unreviewed 数量；
- 各 flag/category/source 的数量；
- 输出 Dataset fingerprint；
- 代码 commit/config hash；
- 失败和失效 operation 数量。

若发现 content hash、block hash 或 operation 边界不匹配，默认整个 materialization 失败，不允许静默跳过后继续生成“看似成功”的数据集。

Materialize 输出已经是人看到并确认的最终正文，只运行 validation-only check，然后直接进入 `build_bin.py`。

---

## 10. 超大数据集约束

- Importer 必须迭代式读取和分批写 Arrow。
- HF Dataset 使用 memory mapping，后端按 row 读取。
- 每个后端 worker 缓存打开的 Dataset handle，不在每个请求重新 `load_from_disk()`。
- API 返回单 Document 或有限 Blocks，不返回整份 Dataset。
- Queue 使用 ordinal/cursor，不使用百万级 SQLite OFFSET。
- materialize 流式遍历和分片写出，不构造 Python 全量 list。
- 所有长任务写进度、吞吐量、当前 source 和预计输出量。
- 临时输出与最终输出同盘，成功后原子替换。
- 任何“支持大 JSON Array”的实现必须真正流式解析。

---

## 11. 测试要求

在做 UI 前先用 CLI 跑通以下不变量：

- 相同输入、adapter version 和 mapping 生成相同 doc_id。
- 重复正文但不同 source locator 生成不同 doc_id。
- source 内容改变后旧 annotation 不会被套用。
- Import 中断不会留下可见的半成品 revision。
- Block parser 对同一文本和版本完全确定。
- emoji 前后的 code point span 能正确删除。
- 重叠 Span operation 被拒绝。
- current state 与 Event Log 在事务失败时不会分叉。
- undo 产生补偿事件并恢复状态。
- 固定 event snapshot 多次 materialize 结果 hash 相同。
- `keep_only/drop_rejected` 对 unreviewed 的行为符合 manifest。
- 输出 Dataset 经 tokenizer/build_bin 后文档边界和 EOS 正确。

---

## 12. MVP 路线

### V0.1：快速审核闭环

Backend：

- [x] Source Manifest、Canonical Schema 和目录布局。
- [x] HF local、JSONL、TXT importer。
- [x] Inspect / Preview / Mapping config。
- [x] 分批 Arrow 写入和原子提交。
- [x] Prepare Review Snapshot，并保留 provenance。
- [x] SQLite migration、Project、Queue、Review、Event 表。
- [x] 固定种子随机/来源配额 Queue。
- [x] Document API 和确定性 Block parser。
- [x] Document Keep / Drop / Unsure、quality、notes。
- [x] Primary category、flags 和 guideline version。
- [x] Block Keep / Drop。
- [x] 单步 Undo。
- [x] `keep_only/drop_rejected` Materialize。
- [x] Materialization manifest 和 validation-only check。

Frontend：

- [x] Project/Queue 选择。
- [x] 单 Document + Blocks 查看。
- [x] provenance 面板。
- [x] 快捷键标注和自动前进。
- [x] Block Drop 和最终正文预览。
- [x] Review 保存状态、错误提示和进度。

V0.1 明确不做：

- Span replace/delete；
- Block move/merge/split；
- 通用 JSON 大数组（除非引入真正流式 parser）；
- Browser Extension；
- 搜索引擎和全文索引；
- 多用户权限和协同编辑；
- 自动分类模型；
- 通用爬虫。

### V0.2：精细编辑与数据分析

- Span delete/replace 和 hash/offset 冲突检查。
- Browser Extension、整页/选区 DOM 导入。
- Dataset/来源统计、搜索、过滤和批量操作。
- 近重复检测和 boilerplate 聚类。
- Mapping Template 管理和 Import History。
- Materialization 历史对比。

### V0.3：辅助筛选

- 文本长度、语言比例、URL、重复率、entropy 等特征。
- Logistic Regression / LightGBM baseline。
- 小型中文 Encoder 做 keep/drop、quality、category、flags。
- AI 建议、人工 Accept/Reject。
- 主动学习，同时保留随机 holdout 监控偏差。

---

## 13. 实施顺序

1. 固定 Canonical Schema、Source Manifest 和 ID 算法。
2. 实现 SQLite migration 与纯 Python Block/Materialize 核心。
3. 用小型合成 Dataset 完成确定性和回滚测试。
4. 实现 HF local importer，再实现 JSONL/TXT。
5. 用 CLI 跑通 Import → Prepare → Queue → Review → Materialize → build_bin。
6. 再实现 FastAPI。
7. 最后实现最小 React 审核台。
8. 用真实数据标注一批，观察工作流后再决定 Span 编辑、搜索和模型功能。

不要从 Browser Extension、漂亮 UI 或自动模型开始。第一阶段的完成标准是：任何一条最终训练文本都能回答“来自哪里、经过什么规则、谁在什么版本上做了什么决定、如何确定性重建”。
