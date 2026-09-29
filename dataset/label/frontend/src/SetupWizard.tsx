import { useEffect, useState } from "react";
import { requestJson } from "./api";
import type { Project } from "./types";

type Mapping = {
  text_fields: string[];
  text_separator: string;
  title_field: string | null;
  url_field: string | null;
  local_id_field: string | null;
  metadata_fields: string[];
  record_adapter: string | null;
};

type DownloadSource = {
  source_id: string;
  path: string;
  adapter: "jsonl" | "huggingface_local";
  available: boolean;
  record_count: number | null;
  ready: boolean;
  mapping: Mapping;
};

type CatalogPrepare = {
  revision_directory: string;
  manifest: { prepare_revision: string; output_records: number; changed_records?: number; config: { simplify_chinese?: boolean } };
};

type CatalogSource = {
  revision_directory: string;
  manifest: { source_id: string; source_revision: string; original_location: string; record_count: number };
  prepares: CatalogPrepare[];
};

type AutoJob = {
  status?: "running" | "stopping" | "paused" | "interrupted" | "completed" | "failed" | "needs_retry" | "partial";
  processed?: number;
  attempts?: number;
  source_attempts?: Record<string, number>;
  source_outcomes?: Record<string, { keep?: number; drop?: number; incomplete?: number }>;
  source_activity?: Record<string, { capacity: number; active: number; completed: number;
    incomplete: number; last_seconds: number | null; rate_per_minute: number }>;
  source_errors?: Record<string, string>;
  sources?: Array<{ source: string; model: string; retry_only?: boolean }>;
  completed?: number;
  remaining?: number;
  counts?: { keep?: number; drop?: number; incomplete?: number };
  batch_counts?: { keep?: number; drop?: number; incomplete?: number };
  web_session_cleanup_errors?: string[];
  manual_completed?: number;
  manual_llm_saved?: number;
  manual_unsure?: number;
  export?: { exported_records: number; dataset: string; updated_at: string } | null;
  export_stale?: boolean;
  requests?: number;
  input_tokens?: number;
  output_tokens?: number;
  rate_per_minute?: number;
  eta_seconds?: number | null;
  elapsed_seconds?: number;
  error?: string;
};

type AutoQueue = {
  queue_id: string;
  project_id: string;
  project_name: string;
  queue_name: string;
  source_id: string;
  source_revision: string;
  source_records: number;
  prepared_directory: string;
  queue_records: number;
  job: AutoJob;
};

const count = (value?: number) => (value ?? 0).toLocaleString();
const duration = (seconds?: number | null) => seconds == null ? "—" : `${Math.floor(seconds / 3600)}时${Math.floor(seconds % 3600 / 60)}分`;
const statusName: Record<string, string> = { running: "运行中", stopping: "暂停中", paused: "已暂停", interrupted: "服务中断，可续跑", completed: "已完成", failed: "失败", needs_retry: "有未完成条目", partial: "部分完成" };
const sourceName: Record<string, string> = { deepseek: "DeepSeek API", qwen_api: "千问 API",
  local: "本地 Qwen", deepseek_web: "DeepSeek 网页", qwen_web: "千问网页",
  kimi_web: "Kimi 网页", doubao_web: "豆包网页",
  chatglm_web: "智谱清言网页", spark_web: "讯飞星火网页", wenxin_web: "文心网页", yuanbao_web: "腾讯元宝网页",
  deepseek_api_retry: "DeepSeek API · 失败重试" };

export type SetupPage = "downloads" | "imported" | "batch";

type Props = {
  page: SetupPage;
  onPageChange: (page: SetupPage) => void;
  onBusyChange: (busy: boolean) => void;
  projects: Project[];
  onComplete: (projectId: string, queueId: string) => Promise<void> | void;
};

function pathName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).at(-1) || path;
}

export default function SetupWizard({ page, onPageChange, onBusyChange, projects, onComplete }: Props) {
  const [downloads, setDownloads] = useState<DownloadSource[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [catalog, setCatalog] = useState<CatalogSource[]>([]);
  const [autoQueues, setAutoQueues] = useState<AutoQueue[]>([]);
  const [prepareDirectory, setPrepareDirectory] = useState("");
  const [prepareRecords, setPrepareRecords] = useState<number | null>(null);
  const [selectedName, setSelectedName] = useState("");
  const [projectId, setProjectId] = useState("");
  const [newProjectName, setNewProjectName] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { onBusyChange(busy); }, [busy, onBusyChange]);
  useEffect(() => () => onBusyChange(false), [onBusyChange]);
  const [prepareProgress, setPrepareProgress] = useState<{ processed: number; total: number; stage: string } | null>(null);
  const [message, setMessage] = useState("");
  const selected = downloads.find((source) => source.source_id === selectedId);

  const refreshAutoQueues = async () => {
    setAutoQueues(await requestJson<AutoQueue[]>("/api/auto-clean"));
  };

  const refreshSources = async () => {
    const [sources, imported, queues] = await Promise.all([
      requestJson<DownloadSource[]>("/api/download-sources"),
      requestJson<CatalogSource[]>("/api/catalog"),
      requestJson<AutoQueue[]>("/api/auto-clean"),
    ]);
    setDownloads(sources);
    setSelectedId((current) => sources.some((source) => source.source_id === current)
      ? current : sources.find((source) => source.available && source.ready)?.source_id || "");
    setCatalog(imported);
    setAutoQueues(queues);
  };

  useEffect(() => {
    void refreshSources().catch((error) => setMessage(`读取数据源失败：${String(error)}`));
  }, []);

  useEffect(() => {
    if (page !== "batch") return;
    const timer = window.setInterval(() => {
      void refreshAutoQueues().catch(() => {});
    }, 2000);
    return () => window.clearInterval(timer);
  }, [page]);

  const clearPrepared = () => {
    setPrepareDirectory("");
    setPrepareRecords(null);
    setSelectedName("");
  };

  const prepareSource = async (revisionDirectory: string) => {
    const progressId = crypto.randomUUID();
    setPrepareProgress({ processed: 0, total: 0, stage: "opening" });
    const timer = window.setInterval(() => {
      void requestJson<{ processed: number; total: number; stage: string }>(
        `/api/prepares/progress/${progressId}`,
      ).then(setPrepareProgress).catch(() => {});
    }, 400);
    try {
      const result = await requestJson<{ revision_directory: string; manifest: { output_records: number } }>("/api/prepares", {
        method: "POST",
        body: JSON.stringify({
          source_revision_directory: revisionDirectory, progress_id: progressId,
          config: { simplify_chinese: true }, max_shard_size: "1GB", read_batch_size: 1024,
        }),
      });
      setPrepareProgress({ processed: result.manifest.output_records, total: result.manifest.output_records, stage: "done" });
      return result;
    } finally {
      window.clearInterval(timer);
    }
  };

  const importSelected = async () => {
    if (!selected || !selected.available || !selected.ready) return;
    setBusy(true);
    clearPrepared();
    setPrepareProgress(null);
    try {
      setMessage(`正在导入 ${selected.source_id}；大数据集的进度可在后端终端查看…`);
      const imported = await requestJson<{ revision_directory: string }>("/api/imports", {
        method: "POST",
        body: JSON.stringify({
          source_id: selected.source_id, adapter: selected.adapter, path: selected.path,
          mapping: selected.mapping, max_shard_size: "1GB",
        }),
      });
      setMessage("导入完成，正在逐条转换为简体并生成清洗副本…");
      const prepared = await prepareSource(imported.revision_directory);
      setPrepareDirectory(prepared.revision_directory);
      setPrepareRecords(prepared.manifest.output_records);
      setSelectedName(selected.source_id);
      await refreshSources();
      onPageChange("imported");
      setPrepareProgress(null);
      setMessage(`已导入 ${prepared.manifest.output_records.toLocaleString()} 条。`);
    } catch (error) {
      setPrepareProgress(null);
      setMessage(`导入失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const prepareExisting = async (source: CatalogSource) => {
    setBusy(true);
    try {
      setMessage("正在把已有原始快照转换为简体清洗副本…");
      const prepared = await prepareSource(source.revision_directory);
      setPrepareDirectory(prepared.revision_directory);
      setPrepareRecords(prepared.manifest.output_records);
      setSelectedName(pathName(source.manifest.original_location));
      await refreshSources();
      setPrepareProgress(null);
      setMessage(`已准备 ${prepared.manifest.output_records.toLocaleString()} 条。`);
    } catch (error) {
      setPrepareProgress(null);
      setMessage(`准备数据失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const queueFor = (source: CatalogSource) => autoQueues.find((queue) =>
    queue.source_id === source.manifest.source_id && queue.source_revision === source.manifest.source_revision
    && source.prepares.some((item) => item.manifest.config.simplify_chinese
      && item.revision_directory === queue.prepared_directory));

  const startAuto = async (queueId: string) => {
    setBusy(true);
    try {
      await requestJson(`/api/auto-clean/${queueId}/start`, { method: "POST", body: "{}" });
      await refreshAutoQueues();
      setMessage("自动清洗已启动；离开此页面后仍会继续运行。可在清洗页逐条查看和修改结果。");
    } catch (error) {
      setMessage(`启动失败：${String(error)}`);
    } finally { setBusy(false); }
  };

  const exportAuto = async (queueId: string) => {
    setBusy(true);
    setMessage("正在生成合并数据集，人工确认结果会覆盖对应的自动结果…");
    try {
      const result = await requestJson<{ exported_records: number }>(`/api/auto-clean/${queueId}/export`, { method: "POST" });
      await refreshAutoQueues();
      setMessage(`合并完成：${count(result.exported_records)} 条正文。`);
    } catch (error) {
      setMessage(`合并失败：${String(error)}`);
    } finally { setBusy(false); }
  };

  const stopAuto = async (queueId: string) => {
    setBusy(true);
    try {
      await requestJson(`/api/auto-clean/${queueId}/stop`, { method: "POST" });
      await refreshAutoQueues();
      setMessage("正在等待已发出的请求结束，随后保存进度并暂停。");
    } catch (error) {
      setMessage(`暂停失败：${String(error)}`);
    } finally { setBusy(false); }
  };

  const startCleaning = async (automatic: boolean) => {
    if (!prepareDirectory) return;
    setBusy(true);
    try {
      const source = catalog.find((entry) => entry.prepares.some((item) => item.revision_directory === prepareDirectory));
      const existing = source && autoQueues.find((queue) => queue.prepared_directory === prepareDirectory
        && queue.source_id === source.manifest.source_id && queue.source_revision === source.manifest.source_revision);
      if (existing) {
        if (automatic) await startAuto(existing.queue_id);
        else await onComplete(existing.project_id, existing.queue_id);
        return;
      }
      setMessage("正在创建全量清洗队列…");
      let targetProjectId = projectId;
      if (!targetProjectId) {
        const project = await requestJson<Project>("/api/projects", {
          method: "POST", body: JSON.stringify({ name: newProjectName.trim() || `${selectedName}清洗` }),
        });
        targetProjectId = project.project_id;
        setProjectId(targetProjectId);
      }
      await requestJson(`/api/projects/${targetProjectId}/sources`, {
        method: "POST", body: JSON.stringify({ prepare_revision_directory: prepareDirectory }),
      });
      const queue = await requestJson<{ queue_id: string }>(`/api/projects/${targetProjectId}/queues`, {
        method: "POST", body: JSON.stringify({ name: "全量清洗", policy: "full_dataset" }),
      });
      if (automatic) {
        await startAuto(queue.queue_id);
      } else await onComplete(targetProjectId, queue.queue_id);
    } catch (error) {
      setMessage(`开始清洗失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  return <section className="setup-wizard">
    <div className="setup-toolbar">
      <h2>{page === "downloads" ? "导入数据" : page === "imported" ? `已导入 (${catalog.length})` : "批量清洗"}</h2>
    </div>

    {page === "downloads" ? <>
      <div className="setup-download-list">
        {downloads.map((source) => <button type="button" key={source.source_id}
          className={`setup-download-item ${selectedId === source.source_id ? "selected" : ""}`}
          disabled={busy || !source.available || !source.ready}
          onClick={() => { setSelectedId(source.source_id); clearPrepared(); setMessage(""); }}>
          <strong>{source.source_id}</strong>
          <span>{!source.available ? "尚未下载" : !source.ready ? "无法识别字段" : `${source.adapter === "jsonl" ? "JSONL" : "数据集"}${source.record_count !== null ? ` · ${source.record_count.toLocaleString()} 条` : ""}`}</span>
        </button>)}
        {!downloads.length && <p className="setup-empty">下载目录没有可用数据源。</p>}
      </div>
      <div className="setup-actions">
        <button type="button" className="primary" disabled={busy || !selected?.ready}
          onClick={() => void importSelected()}>导入{selected ? ` ${selected.source_id}` : "数据"}</button>
      </div>
    </> : page === "imported" ? <div className="setup-catalog">
      {catalog.map((source) => {
        const simplified = source.prepares.filter((item) => item.manifest.config.simplify_chinese);
        const queue = queueFor(source);
        return <article className="setup-catalog-item" key={source.manifest.source_revision}>
          <div className="setup-catalog-info"><strong>{pathName(source.manifest.original_location)}</strong>
            <span>原始 {count(source.manifest.record_count)} 条 · {simplified.length ? `简体副本 ${count(simplified.at(-1)?.manifest.output_records)} 条` : "尚无简体副本"}
              {simplified.length > 0 && <> · 繁简变更 {count(simplified.at(-1)?.manifest.changed_records)} 条</>}</span>
          </div>
          <div className="setup-catalog-actions">
            {!simplified.length && <button type="button" className="secondary" disabled={busy}
              onClick={() => void prepareExisting(source)}>生成简体副本</button>}
            {!queue && simplified.map((item) => <button type="button" key={item.manifest.prepare_revision}
              className="secondary" disabled={busy} onClick={() => {
                setPrepareDirectory(item.revision_directory);
                setPrepareRecords(item.manifest.output_records);
                setSelectedName(pathName(source.manifest.original_location));
                setMessage("");
                onPageChange("batch");
              }}>用于批量清洗</button>)}
            {queue && <>
              <button type="button" className="secondary" disabled={busy}
                onClick={() => void onComplete(queue.project_id, queue.queue_id)}>逐条查看</button>
              <button type="button" className="secondary" disabled={busy}
                onClick={() => onPageChange("batch")}>查看批量进度</button>
            </>}
          </div>
        </article>;
      })}
      {!catalog.length && <p className="setup-empty">还没有已导入的数据。</p>}
    </div> : <div className="setup-catalog">
      {catalog.filter((source) => source.prepares.some((item) => item.manifest.config.simplify_chinese)).map((source) => {
        const simplified = source.prepares.filter((item) => item.manifest.config.simplify_chinese);
        const queue = queueFor(source);
        const job = queue?.job || {};
        const total = queue?.queue_records || source.manifest.record_count;
        const complete = job.completed || 0;
        const active = job.status === "running" || job.status === "stopping";
        return <article className="setup-catalog-item" key={source.manifest.source_revision}>
          <div className="setup-catalog-main">
            <div className="setup-catalog-info"><strong>{pathName(source.manifest.original_location)}</strong>
              <span>原始 {count(source.manifest.record_count)} 条 · {simplified.length ? `简体副本 ${count(simplified.at(-1)?.manifest.output_records)} 条` : "尚无简体副本"}
                {simplified.length > 0 && <> · 繁简变更 {count(simplified.at(-1)?.manifest.changed_records)} 条</>}</span>
            </div>
            {queue && <div className="setup-auto-progress">
              <div className="setup-auto-progress-head"><strong>{statusName[job.status || ""] || "未开始自动清洗"}</strong>
                <span>完成 {count(complete)} / {count(total)} · {total ? Math.round(10000 * complete / total) / 100 : 0}%</span></div>
              <progress value={complete} max={total || 1} />
              <div className="setup-auto-stats">
                <span>保留 {count(job.counts?.keep)}</span><span>丢弃 {count(job.counts?.drop)}</span>
                <span>未完成 {count(job.counts?.incomplete)}</span><span>待处理 {count(job.remaining ?? total)}</span>
                <span>人工确认 {count(job.manual_completed)}</span><span>单条 LLM 保存 {count(job.manual_llm_saved)}</span>
                <span>人工待定 {count(job.manual_unsure)}</span><span>批量已处理 {count((job.batch_counts?.keep || 0) + (job.batch_counts?.drop || 0) + (job.batch_counts?.incomplete || 0))}</span>
                <span>批量完成速度 {count(Math.round(job.rate_per_minute || 0))} 条/分</span>
                <span>预计剩余 {duration(job.eta_seconds)}</span><span>API 请求 {count(job.requests)}</span>
              </div>
              {!!job.sources?.length && <section className="setup-source-status" aria-label="各模型来源工作情况">
                <h3>模型工作情况</h3>
                <div className="setup-source-list">{job.sources.map((item) => {
                  const activity = job.source_activity?.[item.source];
                  const outcomes = job.source_outcomes?.[item.source];
                  const error = job.source_errors?.[item.source];
                  const state = error ? "已停用" : job.status === "running" || job.status === "stopping"
                    ? activity?.active ? "处理中" : item.retry_only ? "等待失败条目" : "等待任务" : "未运行";
                  return <div className="setup-source-item" key={item.source}>
                    <div className="setup-source-heading"><strong>{sourceName[item.source] || item.source}</strong>
                      <span className={error ? "error" : ""}>{state}</span></div>
                    <small>{item.model}</small>
                    <div className="setup-source-metrics">
                      <span>活跃 {activity ? `${activity.active}/${activity.capacity}` : "—"}</span>
                      <span>本轮完成 {count(activity?.completed)}</span>
                      <span>本轮未完成 {count(activity?.incomplete)}</span>
                      <span>近一分钟 {activity ? activity.rate_per_minute.toFixed(1) : "—"} 条/分</span>
                      <span>累计尝试 {count(job.source_attempts?.[item.source])}</span>
                      {outcomes && <><span>累计完成 {count((outcomes.keep || 0) + (outcomes.drop || 0))}</span>
                        <span>累计未完成 {count(outcomes.incomplete)}</span></>}
                    </div>
                    {error && <p className="error">{error}</p>}
                  </div>;
                })}</div>
              </section>}
              <details><summary>更多统计与输出</summary>
                <p>队列：{queue.queue_name}（{queue.project_name}） · {count(queue.queue_records)} 条 · 已扫描 {count(job.processed)} 条 · 共尝试 {count(job.attempts)} 次</p>
                <p>输入 {count(job.input_tokens)} tokens · 输出 {count(job.output_tokens)} tokens · 已运行 {duration(job.elapsed_seconds)}</p>
                {job.sources?.length ? <p>本轮模型：{job.sources.map((item) => `${sourceName[item.source] || item.source}（${item.model}）`).join("、")}</p> : null}
                {Object.keys(job.source_attempts || {}).length ? <p>模型日志尝试：{Object.entries(job.source_attempts || {}).map(([source, attempts]) => `${sourceName[source] || source} ${count(attempts)} 次`).join(" · ")}</p> : null}
                <p>累计尝试按进度日志记录模型结果，包含失败和重试；它不等于完成条数，也不等于实际 HTTP 请求数。</p>
                {Object.keys(job.source_errors || {}).length ? <p className="error">本轮已停用：{Object.entries(job.source_errors || {}).map(([source, error]) => `${source}（${error}）`).join("；")}</p> : null}
                {job.web_session_cleanup_errors?.length ? <p className="error">网页会话删除失败：{job.web_session_cleanup_errors.join("；")}</p> : null}
                <p>总完成数已按队列条目去重；人工确认与批量已处理可能是同一条。暂停会先停止派发，再等待当前模型请求写入。</p>
                <p>合并语料：{job.export ? `${count(job.export.exported_records)} 条${job.export_stale ? " · 有新修改，需重新生成" : " · 已更新"}` : "尚未生成"}。人工结果优先，保存在服务端的 batch_cleaned/{queue.queue_id}/ 目录。</p>
                {job.error && <p className="error">{job.error}</p>}
              </details>
            </div>}
          </div>
          <div className="setup-catalog-actions">
            {!queue && simplified.map((item) => <button type="button" key={item.manifest.prepare_revision} disabled={busy}
              className={prepareDirectory === item.revision_directory ? "selected" : "secondary"}
              onClick={() => {
                setPrepareDirectory(item.revision_directory);
                setPrepareRecords(item.manifest.output_records);
                setSelectedName(pathName(source.manifest.original_location));
                setMessage("");
              }}>{prepareDirectory === item.revision_directory ? "已选" : "选用"}</button>)}
            {queue && <button type="button" className="secondary" disabled={busy}
              onClick={() => void onComplete(queue.project_id, queue.queue_id)}>逐条查看</button>}
            {queue && <button type="button" className="secondary" disabled={busy}
              onClick={() => void exportAuto(queue.queue_id)}>{job.export ? "重新生成合并语料" : "生成合并语料"}</button>}
            {queue && simplified.length > 0 && (active
              ? <button type="button" className="secondary" disabled={busy || job.status === "stopping"}
                  onClick={() => void stopAuto(queue.queue_id)}>暂停</button>
              : <button type="button" className="primary" disabled={busy || job.status === "completed"}
                  onClick={() => void startAuto(queue.queue_id)}>{job.status ? "继续自动清洗" : "开始自动清洗"}</button>)}
          </div>
        </article>;
      })}
      {!catalog.some((source) => source.prepares.some((item) => item.manifest.config.simplify_chinese)) &&
        <p className="setup-empty">还没有可清洗的简体副本，请先在已导入页面准备数据。</p>}
    </div>}

    {message && <div className={`setup-message ${message.includes("失败") ? "error" : ""}`} role="status" aria-live="polite">
      {busy && <span className="spinner" />}{message}
    </div>}
    {prepareProgress && <div className="setup-progress" role="status">
      <span>繁体转简体：{prepareProgress.total ? `${prepareProgress.processed.toLocaleString()} / ${prepareProgress.total.toLocaleString()} 条 · ${Math.round(100 * prepareProgress.processed / prepareProgress.total)}%` : "正在读取数据集…"}</span>
      <progress value={prepareProgress.total ? prepareProgress.processed : undefined} max={prepareProgress.total || 1} />
    </div>}

    {page === "batch" && prepareDirectory && !autoQueues.some((queue) => queue.prepared_directory === prepareDirectory) && <div className="setup-launch">
      <div className="setup-ready-dataset"><strong>{selectedName}</strong><span>{prepareRecords?.toLocaleString() ?? "—"} 条 · 简体副本已就绪</span></div>
      <div className="setup-project-row">
        <label>项目<select value={projectId} disabled={busy} onChange={(event) => setProjectId(event.target.value)}>
          <option value="">新建项目</option>{projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}</select></label>
        {!projectId && <label>项目名称<input value={newProjectName} disabled={busy}
          onChange={(event) => setNewProjectName(event.target.value)} placeholder={`${selectedName || "语料"}清洗`} /></label>}
        <button type="button" className="secondary" disabled={busy} onClick={() => void startCleaning(false)}>逐条查看</button>
        <button type="button" className="primary" disabled={busy} onClick={() => void startCleaning(true)}>开始自动清洗</button>
      </div>
    </div>}
  </section>;
}
