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
  → dataset/data_pipeline/build_bin.py
```

正文保存在 HF Dataset/Arrow，SQLite 只保存项目、全量清洗任务、实际标注状态和 append-only event；不会把千万条正文或待办索引塞进数据库。

## 启动清洗台

在项目根目录运行：

```bash
./clean
```

然后在本机打开 `http://127.0.0.1:8000`；同一局域网的设备可使用本机的局域网 IP，例如 `http://192.168.1.3:8000`。`./clean` 默认监听所有网卡，也可用 `CLEAN_HOST=127.0.0.1 ./clean` 仅供本机访问。前端已有 `dist/` 时，只运行这一条命令即可；服务已经运行时脚本会直接提示地址。按 Ctrl+C 停止服务。

需要修改前端时，再另开终端运行：

```bash
cd dataset/label/frontend
npm run dev
```

开发页面地址是 `http://127.0.0.1:5173`。修改完可运行 `npm run build` 更新由后端托管的 `dist/`。

默认数据根目录是 `dataset/label/data`，可通过 `LABEL_DATA_ROOT` 修改。

## LLM 自动清洗

网页点击 **LLM 清洗本条** 后，完整清洗结果直接写入当前文档的正文审核记录，删除的片段不会留在训练正文里。页面只显示保存状态和删改数量，右侧仍可对照原文。无效的局部改写会被忽略，不会使同一块的有效删除失效；旧报告中的失败块会用已保存的模型响应恢复，必要时才重新调用模型。仍有未完成分块的文档不会自动保存为干净正文。

在网页的「导入数据 → 已导入」中，每个来源显示原始条数、简体副本条数和繁简变更条数。创建全量队列后，可直接点击「开始自动清洗」。已有队列显示人工确认、单条 LLM 保存、人工待定和批量处理数量；保留/丢弃/未完成及完成百分比按同一队列条目去重，每 2 秒刷新。点击「暂停」立即停止派发新文档，等待正在执行的文档完成当前模型调用和结果落盘，然后状态变为「已暂停」；它不会中断已经发出的 HTTP 请求，也不影响单条人工清洗。再次点击「继续自动清洗」从进度文件续跑。关闭浏览器不会暂停后端任务；重启服务后，旧任务显示「服务中断，可续跑」。任务不会自动在导入完成时发起远程调用。

单条清洗成功后保存在项目的 SQLite 审核记录；批量清洗的逐条日志保存在输出目录。进度接口把两者按队列位置合并，因此手动清洗已在批量范围内的条目会显示在“单条 LLM 保存”中，但总完成数不会重复加一。旧版翻页曾自动保存“待定”审核；现在未修改的翻页不会写入新审核，历史待定仍保留在审核记录中且不计入已完成。

已勾选的批量模型仍未完成、且该条从未触发内容敏感拒绝时，才交给 DeepSeek 付费 API 再尝试一次；该 API 不领取正常新条目。已有未完成条目在续跑时也适用，失败重试及用量会分别记入进度；同一条不会反复自动付费重试。设置页可关闭这一补救开关；没有配置 DeepSeek API Key 时也不会调用付费 API。千问网页整源验证拒绝产生的未完成记录会先由已勾选来源接手，不自动大量转交付费 API。自动任务按设置页勾选的模型混合派发：本地模型 1 个并发槽、DeepSeek 网页 1 个、千问网页 1 个、Kimi 网页 1 个、豆包网页 1 个、每个付费 API 6 个；快来源在槽位空出后继续领取下一条。总文档并发和请求并发仍可通过命令参数限制。某个来源出现 401、403 或 429 时，本轮停用该来源并将未完成条目交给其他已选来源。读取数据和人工标注状态在主线程完成；长文自身可并发分块，但仍受全局 API 上限约束。批量结果完成后立即写入 `cleaned.jsonl` 和 `progress.jsonl`，避免慢模型阻塞其他来源；这两个中间文件按完成顺序记录，最终 `effective_cleaned.jsonl` 仍按队列顺序导出。`cleaned.jsonl` 只包含批量路径已完成且保留的正文。`progress.jsonl` 按条记录状态、序号、耗时、字符数和模型用量；`manifest.json` 保存持续更新的汇总、速度和任务状态；各来源的“累计尝试”是写入进度日志的结果次数，包含失败和重试，不是成功条数或实际 HTTP 请求数。写入时先同步正文，再同步进度，重启会截掉未记录的正文尾部。未完成条目不进入清洗结果，下次续跑优先重试。人工明确保留或丢弃的条目直接复用，不再向模型发送请求。人工页按相同队列序号直接显示批量保留、丢弃和未完成结果；批量与单条共用同一个 `LlmCleaner.clean()` 及完成判定，批量每篇只调用一次，未完成的条目留待续跑。页面的「生成合并语料」会按队列顺序输出 `effective_cleaned.jsonl`：人工确认优先于先前的批量结果；每次人工修改或批量进度更新后，页面会标记旧合并文件需要重新生成。全量任务完成时会自动生成一次。输出目录按队列、模型和提示词版本隔离。

批量清洗当前 Wiki 队列也可在项目根目录运行：

```bash
./clean-batch 100
```

参数是本次再处理的条数；省略参数则持续处理整个队列。中断后再次运行会从进度文件继续，不重复处理已完成条目；未完成条目会重试，重试成功后的文本追加到输出末尾，因此输出行顺序可能与原始队列不同。输出在 `dataset/label/data/batch_cleaned/wiki_zh_20231101_full/deepseek-deepseek-flash-prose_extraction_v6/cleaned.jsonl`，每行只有 `text` 字段；它是批量原始产物。如果批量处理后又手动修改了条目，使用合并后的 `effective_cleaned.jsonl` 作为最终数据集。整篇删除或仍有未完成分块的文档不会进入该文件；数量见 `manifest.json`，逐条状态见 `progress.jsonl`。已有人工确认并保存的正文会直接使用，避免重复调用模型。全量队列有 1,384,748 条，远程模型全量处理需要大量时间与 API 费用，可分批运行。

通用入口是 `python -m dataset.label.cli llm-clean-batch --queue-id <queue_id> [--limit N]`，每次结束也会生成当前的合并语料；可用 `--workers`、`--max-requests` 调整并发，也可指定 `--data-root` 和 `--output-directory`。网页与 CLI 指向同一进度目录时会用任务锁避免同时写入。

服务配置在 `configs/label.json`，当前使用 DeepSeek 的 `deepseek-flash`。如需切回 DashScope，把 `provider` 改为 `dashscope`，并将 `model`、`base_url` 改为原来的 Qwen 配置。后端自动读取**项目根目录**的 `.env`（不受启动目录影响），系统环境变量优先：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek API密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
# DashScope 模式使用 DASHSCOPE_API_KEY、DEFAULT_MODEL 和可选的 DASHSCOPE_BASE_URL
```

`.env` 已加入 Git 忽略；可参考根目录 `.env.example`。依赖中新增 `python-dotenv`，已有环境执行 `python -m pip install python-dotenv`。修改配置或 `.env` 后重启后端。

远程模式只调用 `/chat/completions`，使用 Bearer 认证、关闭思考和 JSON 输出模式；Schema 放在提示词中，返回后仍执行原有 Pydantic、片段编号及只删不改写校验。DeepSeek 的接口与关闭思考参数依据 [Chat API](https://api-docs.deepseek.com/api/create-chat-completion/) 和 [思考模式文档](https://api-docs.deepseek.com/guides/thinking_mode/)；DashScope 模式依据 [百炼 Chat API 文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)。密钥仅由后端读取，不进入前端或建议报告。鉴权、限流、额度及连接错误会显示在原有工作区中。

如需切回原有 4096 上下文的本地 llama.cpp，将 `provider` 设为 `llamacpp`、`base_url` 设为 `http://127.0.0.1:8080`、`model` 设为本地模型名，并恢复 `context_tokens=4096`、`max_output_tokens=1536`。该模式保留 `/props`、`/apply-template`、`/tokenize` 和 `/v1/chat/completions` 适配，且只允许回环地址，不读取远端模型环境配置。

编辑器与批量清洗按每次换行分块；连续空行保留为相邻块之间的分隔符，删除单行后不会额外插入空行。模型还会把单行内的长句拆成更小的判断片段；成对中文引号内的问号和句号不作为普通句末切分点。超长引文仍受单片长度和上下文上限约束。远端最多 4 块并发处理、本地模型默认串行；结果仍按原文顺序组装，不静默截断。远程模式按包含 Schema 的完整提示词 UTF-8 字节数保守估算输入预算，预留输出及重试空间，不依赖本地 tokenizer；实际 token 用量以 API 返回值为准。本地模式仍使用服务的聊天模板与 tokenizer 计算。DeepSeek API 当前配置允许单篇最多 100 万字符、单请求最多 30 万字符；仍按片段数和实际上下文预算分块，单篇最多 64 块。其他来源保留各自的 10 万字符上限。超限会明确报错，且单篇超限不会停用整个模型来源。

模型响应和操作记录保存在 `dataset/label/data/llm_suggestions/`，供断点恢复和审计；清洗正文以审核记录或批量输出中的 `text` 为准。

本次工作区只导入 `dataset/data_pipeline/data/downloads/wiki_zh_20231101`，项目为 `wiki_zh_20231101`，全量人工队列为 `wiki_zh_20231101_full`。正文映射 `text`，`title`、`url`、`id` 保存为来源信息。队列包含全部 1,384,748 条文档；网页单条清洗会直接保存；批量清洗使用 `./clean-batch`。

Wiki 队列当前使用全量 `t2s` 后的 Review Snapshot，Prepare 修订为 `62dd96394dd8ffedc9145daa527cbc64`。与旧 Review Snapshot 相比，1,058,168 条正文和 526,266 条标题发生变化，文档身份与顺序保持一致。原始 Raw Snapshot、旧 Review Snapshot 和下载数据集均保留；本次只切换了清洗队列的快照路径。转换配置是 `PrepareConfig(simplify_chinese=True)`，实际处理版本和结果保存在该修订的 `manifest.json`。

网页左侧选择“导入数据”后：

1. 从 `configs/data_pipeline.json` 配置的项目下载目录中选择来源。当前支持 MiniMind JSONL，以及 FineWeb、FineWiki、中文维基和 TigerResearch 的本地 Hugging Face 数据集；不用填写路径。
2. 系统用来源结构推断正文、标题和 ID；页面只列出来源名称、格式和记录数。
3. 点击“导入”。Import 保留原始 Raw Snapshot；随后的 Prepare 对所有正文和标题执行一次繁体转简体。页面显示已处理条数和百分比。旧 Raw Snapshot 也可在“已有导入”中生成简体副本。
4. 选择或新建项目，创建全量清洗队列，然后选择「人工查看」或「开始自动清洗」。标注页直接读取已转换的 Review Snapshot，不再进行繁简转换，也不再显示转换按钮。

正式导入仍须完整读取数据集，118GB 级来源可能需要较长时间。原始下载文件保持不变；简体正文写入版本化的 Review Snapshot。默认 `PrepareConfig(simplify_chinese=True)`，旧版未转简体的快照不会被导入页直接选为清洗数据。

内容适配复用 `dataset/data_pipeline/record_adapters.py`：文章、分类正文、instruction/input/output、问答、问答含 think/reasoning、ShareGPT、role/content 对话（含 messages）、贴吧主楼与回复。只抽取并拼接原有文字，不生成 SFT/DPO 格式；role、分类标签不会混入正文。自定义映射仍支持点路径字符串列；直接选中数组或对象会报错，避免只留下标题而静默丢失正文。

API/CLI 的 mapping 可指定 `record_adapter` 和空 `text_fields`；原有纯字段映射不需修改，旧快照的版本标识保持兼容。这里只做内容格式适配，Import 不执行旧预处理的过滤规则。

已经导入的来源只显示在“已导入”，不再出现在“下载目录”；刷新网页后可以直接继续。

## CLI

查看所有命令：

```powershell
.\.venv\Scripts\python.exe -m dataset.label.cli --help
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

Materialize 输出仍保留 canonical provenance 列，但仅使用人工审核事件；自动任务的最新合并结果在 `effective_cleaned.jsonl` 中。两种输出不能互换。完整运行链与这次代码复查见 [标注工具代码复查](../../docs/label-tool-code-audit-20260928.md)。`dataset/data_pipeline/build_bin.py` 只在检测到标注输出专用的 `dataset.json` 时启用新 loader；原有 `save_to_disk` 预处理结果继续走原来的 `datasets.load_from_disk`，并继续要求只有 `text` 一列。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s dataset\label\tests -v
cd dataset\label\frontend
npm test
npm run build
```

## 清洗模型设置

清洗台左侧“设置”给单条清洗选一个模型，给批量清洗勾选多个来源：DeepSeek API、Qwen API、本地 Qwen 27B、本地 Laya 分类器（实验），以及 DeepSeek、千问、Kimi、豆包、智谱清言、讯飞星火、文心、腾讯元宝的网页来源。默认单条使用 DeepSeek API，批量只勾选本地模型；设置页还有独立的“普通清洗失败后交给 DeepSeek API”开关，修改后在下次启动或续跑批量任务时生效。密钥和网页登录令牌由根目录 `.env` 读取。切换批量模型需先暂停运行中的任务，已有队列进度会继续使用，不会重新处理已完成条目。较快来源会自动接更多条；某来源出现非敏感未完成时，同一条最多再交给另一个勾选来源一次；仍未完成时，再由 DeepSeek API 补一次。明确的内容敏感拒绝留待人工查看，不继续向其他来源派发。尝试都会写入进度。

启动 `./clean` 只打开清洗台，不会占用显卡加载 Qwen 27B；实际选择本地 Qwen 执行单条或批量清洗时，才检测同级 `qwen/` 项目的 CUDA 服务并按需启动。由清洗台启动的服务在退出清洗台后释放。`./clean-batch` 选择本地 Qwen 时也会自动启动并等待就绪，日志位于忽略版本控制的 `dataset/label/data/local_model.log`。Laya 使用项目内的权重和 Python 推理，不另起服务；当前实验选项加载针对维基语料微调的 `dataset/label/models/laya_wiki_cleaning_v1/`，不自动加入默认批量模型。基座误删问题和微调后的对照分别见 [基座评估](../../docs/laya-local-classifier-evaluation.md)与[微调报告](../../docs/laya-finetuning-evaluation-20260930.md)；其他数据源需先抽样验证。

DeepSeek 网页来源复用 [Node 示例](../../deepseek-web-api/README.md) 调用网页内部接口，需要 Node.js 18 或更新版本；在 VS Code Remote 环境下会自动查找其自带的 Node。先从已登录的网页取得自己的令牌，在根目录 `.env` 中添加 `DEEPSEEK_WEB_TOKEN=...`，然后重启 `./clean`；这不是正式的 `DEEPSEEK_API_KEY`。设置页会显示令牌是否已配置。首次请求创建会话，后续单条与批量复用同一 `chat_session_id`，每次请求仍重新解 PoW；各文档使用空 `parent_message_id` 独立判断，避免前一条正文进入后一条的上下文。会话 ID 保存在数据根目录的 `deepseek_web_session.json`，重启后继续使用；深度思考和联网搜索始终关闭。单条和批量仍使用同一套提示词、只删不改写校验与进度记录。网页接口没有可靠的 token 用量回传，因此此来源的用量统计为 0。批量中 DeepSeek 网页固定使用一个会话槽，分别反复处理不同文档；与单条清洗的会话分开。首条请求失败后也保存已创建的会话 ID，避免下一条再次创建。批量任务结束、暂停或失败后会删除它创建的网页会话；单条会话保留以便下次复用。删除失败会在批量统计里提示，并保留本地会话 ID 供下次重试。

千问网页也复用同一清洗器，凭据由 `.env` 的 `QWEN_WEB_AUTH_FILE` 指向 [网页抓取脚本](../../deepseek-web-api/README.md) 导出的 `.qwen-web-auth.json`。批量固定使用一个会话槽，并在上次回复结束后至少等待 20 秒再发下一条；文档间用首轮请求隔离上下文，关闭深度思考和联网搜索；每次请求消耗一份网页登录签名材料；余量低或将过期时，会在凭据文件锁内自动向网页补充，补充失败且旧材料不可用时才停用该来源。网页返回应用层验证拒绝时也会停用本轮千问，其他勾选来源继续处理。批量结束时删除会话。网页接口不返回可比的 token 用量，统计中的 tokens 为 0。

Kimi 网页和豆包网页分别使用 `deepseek-web-api/kimi_chat.mjs`、`deepseek-web-api/doubao_chat.py`；先运行对应的 `kimi_capture_auth.py`、`doubao_capture_auth.py` 导出网页登录凭据，并在根目录 `.env` 设置 `KIMI_WEB_AUTH_FILE`、`DOUBAO_WEB_AUTH_FILE`。豆包还需要同目录的 `doubao_request_template.json`。两者固定各一个批量会话槽；Kimi 关闭深度思考，豆包开启深度思考，两者均关闭联网搜索，复用会话并在批量停止时删除；仍由同一个清洗器负责提示词、结果校验和进度记录。网页接口可能变化，首次启用前可先用单条清洗确认凭据和响应格式。 智谱清言、讯飞星火、文心、腾讯元宝分别复用同目录的 Python 聊天脚本与结构化适配器，凭据文件按 `CHATGLM_WEB_AUTH_FILE`、`SPARK_WEB_AUTH_FILE`、`WENXIN_WEB_AUTH_FILE`、`YUANBAO_WEB_AUTH_FILE` 配置；每个来源一个批量会话槽。

各网页来源独立计时，互不等待；同一来源的人工单条与批量请求互斥，一次只清洗一条数据。千问和 DeepSeek 网页在上次回复结束后至少等待 20 秒，其余网页来源至少等待 15 秒；失败回复也会等待。遇到限流只将该来源间隔加倍，最高 5 分钟；持续成功后才缓慢缩短。网页请求等待限速时不占用正式 API 的并发名额。401、403、429、禁言和签名失效会停派该来源，本轮未完成条目转交其他可用来源。网页接口没有公开稳定的批量额度；这些间隔是保守的运行设置，不能保证账号不会再次被限流。

本地样本效果见 [27B 清洗试验](../../docs/local-qwen-cleaning-evaluation-20260928.md)。
