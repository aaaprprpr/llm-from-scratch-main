import { useEffect, useMemo, useState } from "react";
import { requestJson } from "./api";
import type { Project } from "./types";

type Inspection = {
  source_type: string;
  input_location: string;
  input_fingerprint: string;
  fields: string[];
  record_count: number | null;
  details: Record<string, unknown>;
};

type PreviewDocument = {
  stable_locator: string;
  text: string;
  source_local_id: string | null;
  title: string | null;
  url: string | null;
};

type ImportResult = {
  revision_directory: string;
  reused_existing: boolean;
  source_manifest: {
    source_id: string;
    source_revision: string;
    record_count: number;
  };
};

type PrepareResult = {
  revision_directory: string;
  reused_existing: boolean;
  manifest: {
    source_id: string;
    source_revision: string;
    output_records: number;
    changed_records: number;
  };
};

type CatalogPrepare = {
  revision_directory: string;
  manifest: PrepareResult["manifest"] & { prepare_revision: string };
};

type CatalogSource = {
  revision_directory: string;
  manifest: ImportResult["source_manifest"] & { license: string };
  prepares: CatalogPrepare[];
};

type Props = {
  projects: Project[];
  initialProjectId: string;
  onComplete: (projectId: string, queueId: string) => Promise<void> | void;
  onClose: () => void;
};

function commaValues(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function guessedTextFields(fields: string[]): string[] {
  const preferred = ["title", "text", "content", "body"];
  const selected = preferred.filter((field) => fields.includes(field));
  if (selected.includes("text")) return selected.filter((field) => field !== "content");
  return selected.length ? selected : fields.slice(0, 1);
}

export default function SetupWizard({
  projects,
  initialProjectId,
  onComplete,
  onClose,
}: Props) {
  const [adapter, setAdapter] = useState("huggingface_local");
  const [path, setPath] = useState("");
  const [sourceId, setSourceId] = useState("");
  const [license, setLicense] = useState("unknown");
  const [optionsJson, setOptionsJson] = useState("{}");
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [textFields, setTextFields] = useState("");
  const [titleField, setTitleField] = useState("");
  const [urlField, setUrlField] = useState("");
  const [localIdField, setLocalIdField] = useState("");
  const [metadataFields, setMetadataFields] = useState("");
  const [preview, setPreview] = useState<PreviewDocument[]>([]);
  const [imported, setImported] = useState<ImportResult | null>(null);
  const [prepared, setPrepared] = useState<PrepareResult | null>(null);
  const [prepareDirectory, setPrepareDirectory] = useState("");
  const [prepareRecords, setPrepareRecords] = useState<number | null>(null);
  const [catalog, setCatalog] = useState<CatalogSource[]>([]);
  const [projectId, setProjectId] = useState(initialProjectId);
  const [newProjectName, setNewProjectName] = useState("");
  const [queueName, setQueueName] = useState("随机审核样本");
  const [sampleSize, setSampleSize] = useState(1000);
  const [seed, setSeed] = useState(42);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("先检查本地数据源，或者复用已有 Prepare 快照。");

  const loadCatalog = async () => {
    const values = await requestJson<CatalogSource[]>("/api/catalog");
    setCatalog(values);
  };

  useEffect(() => {
    void loadCatalog().catch((error) => setMessage(`读取目录失败：${String(error)}`));
  }, []);

  useEffect(() => {
    if (initialProjectId) setProjectId(initialProjectId);
  }, [initialProjectId]);

  const sourceOptions = () => {
    const value = JSON.parse(optionsJson || "{}");
    if (!value || Array.isArray(value) || typeof value !== "object") {
      throw new Error("Adapter options 必须是 JSON object");
    }
    return value as Record<string, unknown>;
  };

  const mapping = useMemo(() => ({
    text_fields: commaValues(textFields),
    text_separator: "\n\n",
    title_field: titleField || null,
    url_field: urlField || null,
    local_id_field: localIdField || null,
    metadata_fields: commaValues(metadataFields),
  }), [localIdField, metadataFields, textFields, titleField, urlField]);

  const inspectSource = async () => {
    if (!path.trim()) {
      setMessage("请先填写服务器本机的数据集路径。");
      return;
    }
    setBusy(true);
    setMessage("正在检查 schema 和数据指纹…");
    try {
      const result = await requestJson<Inspection>("/api/imports/inspect", {
        method: "POST",
        body: JSON.stringify({ adapter, path, options: sourceOptions() }),
      });
      setInspection(result);
      const guessed = guessedTextFields(result.fields);
      setTextFields(guessed.join(", "));
      setTitleField(result.fields.includes("title") ? "title" : "");
      setUrlField(result.fields.includes("url") ? "url" : "");
      setLocalIdField(
        ["uniqueKey", "id", "doc_id"].find((field) => result.fields.includes(field)) ?? "",
      );
      setPreview([]);
      setMessage(`检查完成：${result.record_count?.toLocaleString() ?? "未知"} 条记录。请确认字段映射。`);
    } catch (error) {
      setMessage(`检查失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const previewSource = async () => {
    if (!mapping.text_fields.length) {
      setMessage("text_fields 不能为空。");
      return;
    }
    setBusy(true);
    setMessage("正在按当前字段映射生成预览…");
    try {
      const values = await requestJson<PreviewDocument[]>("/api/imports/preview", {
        method: "POST",
        body: JSON.stringify({
          adapter,
          path,
          options: sourceOptions(),
          mapping,
          limit: 3,
        }),
      });
      setPreview(values);
      setMessage("预览生成完成。下方文本就是 Raw Snapshot 将保存的正文。");
    } catch (error) {
      setMessage(`预览失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const importAndPrepare = async () => {
    if (!sourceId.trim()) {
      setMessage("source_id 不能为空，例如 tigerresearch_pretrain_zh_v1。");
      return;
    }
    if (!preview.length) {
      setMessage("必须先预览映射结果，避免把错误字段写入大快照。");
      return;
    }
    setBusy(true);
    try {
      setMessage("正在导入 Raw Snapshot；大数据集可能需要较长时间，请保持后端运行…");
      const importResult = await requestJson<ImportResult>("/api/imports", {
        method: "POST",
        body: JSON.stringify({
          adapter,
          path,
          options: sourceOptions(),
          source_id: sourceId,
          license,
          mapping,
          max_shard_size: "1GB",
        }),
      });
      setImported(importResult);
      setMessage("Raw Snapshot 完成；正在生成一进一出的 Review Snapshot…");
      const prepareResult = await requestJson<PrepareResult>("/api/prepares", {
        method: "POST",
        body: JSON.stringify({
          source_revision_directory: importResult.revision_directory,
          config: {},
          max_shard_size: "1GB",
          read_batch_size: 1024,
        }),
      });
      setPrepared(prepareResult);
      setPrepareDirectory(prepareResult.revision_directory);
      setPrepareRecords(prepareResult.manifest.output_records);
      setSampleSize(Math.min(1000, prepareResult.manifest.output_records));
      await loadCatalog();
      setMessage(
        `导入与 Prepare 完成：${prepareResult.manifest.output_records.toLocaleString()} 条；` +
        `${prepareResult.manifest.changed_records.toLocaleString()} 条规范化后发生变化。`,
      );
    } catch (error) {
      setMessage(`导入/Prepare 失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const prepareExisting = async (revisionDirectory: string) => {
    setBusy(true);
    setMessage("正在为已有 Raw Snapshot 生成 Review Snapshot…");
    try {
      const result = await requestJson<PrepareResult>("/api/prepares", {
        method: "POST",
        body: JSON.stringify({
          source_revision_directory: revisionDirectory,
          config: {},
        }),
      });
      setPrepared(result);
      setPrepareDirectory(result.revision_directory);
      setPrepareRecords(result.manifest.output_records);
      setSampleSize(Math.min(1000, result.manifest.output_records));
      await loadCatalog();
      setMessage("Prepare 完成，现在可以挂载到项目并创建审核队列。");
    } catch (error) {
      setMessage(`Prepare 失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const selectPrepared = (value: CatalogPrepare) => {
    setPrepareDirectory(value.revision_directory);
    setPrepareRecords(value.manifest.output_records);
    setSampleSize(Math.min(1000, value.manifest.output_records));
    setMessage(`已选择 ${value.manifest.source_id} 的 Review Snapshot。`);
  };

  const ensureProject = async (): Promise<string> => {
    if (projectId) return projectId;
    if (!newProjectName.trim()) throw new Error("请选择项目或填写新项目名称");
    const project = await requestJson<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name: newProjectName.trim() }),
    });
    setProjectId(project.project_id);
    return project.project_id;
  };

  const createQueue = async (targetProjectId: string) => {
    return requestJson<{ queue_id: string }>(
      `/api/projects/${targetProjectId}/queues`,
      {
        method: "POST",
        body: JSON.stringify({
          name: queueName.trim() || "随机审核样本",
          policy: "uniform_random",
          seed,
          sample_size: sampleSize,
        }),
      },
    );
  };

  const attachAndCreateQueue = async () => {
    if (!prepareDirectory) {
      setMessage("请先完成 Prepare 或从目录中选择已有 Review Snapshot。");
      return;
    }
    setBusy(true);
    try {
      const targetProjectId = await ensureProject();
      setMessage("正在把固定 Review Snapshot 挂载到项目…");
      await requestJson(`/api/projects/${targetProjectId}/sources`, {
        method: "POST",
        body: JSON.stringify({ prepare_revision_directory: prepareDirectory }),
      });
      setMessage("来源已挂载；正在创建固定种子随机队列…");
      const queue = await createQueue(targetProjectId);
      setMessage("项目和审核队列已就绪。");
      await onComplete(targetProjectId, queue.queue_id);
    } catch (error) {
      setMessage(`项目初始化失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const createQueueForAttachedSources = async () => {
    setBusy(true);
    try {
      const targetProjectId = await ensureProject();
      const queue = await createQueue(targetProjectId);
      setMessage("已为项目当前挂载的所有来源创建随机队列。");
      await onComplete(targetProjectId, queue.queue_id);
    } catch (error) {
      setMessage(`创建队列失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="setup-wizard">
      <div className="setup-heading">
        <div>
          <p className="eyebrow">DATASET ONBOARDING</p>
          <h2>导入数据并建立审核队列</h2>
        </div>
        <button className="ghost" onClick={onClose}>返回审核台</button>
      </div>

      <div className={`setup-message ${message.includes("失败") ? "error" : ""}`}>
        {busy && <span className="spinner" />}
        {message}
      </div>

      <div className="setup-grid">
        <div className="setup-column">
          <div className="setup-step">
            <span className="step-number">1</span>
            <div>
              <h3>检查本地数据源</h3>
              <p>路径是运行 FastAPI 的这台机器上的路径。不会由浏览器上传几十 GB 文件。</p>
            </div>
          </div>
          <label>数据格式
            <select value={adapter} onChange={(event) => setAdapter(event.target.value)} disabled={busy}>
              <option value="huggingface_local">Hugging Face save_to_disk</option>
              <option value="jsonl">JSONL</option>
              <option value="text">TXT</option>
            </select>
          </label>
          <label>本机路径
            <input
              value={path}
              onChange={(event) => setPath(event.target.value)}
              placeholder="data_pipeline/data/downloads/tigerresearch_pretrain_zh"
              disabled={busy}
            />
          </label>
          <div className="two-fields">
            <label>source_id
              <input value={sourceId} onChange={(event) => setSourceId(event.target.value)} placeholder="tigerresearch_pretrain_zh_v1" disabled={busy} />
            </label>
            <label>License
              <input value={license} onChange={(event) => setLicense(event.target.value)} disabled={busy} />
            </label>
          </div>
          <details className="advanced-options">
            <summary>Adapter options</summary>
            <textarea value={optionsJson} onChange={(event) => setOptionsJson(event.target.value)} rows={3} disabled={busy} />
            <p>TXT 示例：{`{"document_strategy":"blank_line_separator"}`}</p>
          </details>
          <button className="primary" onClick={() => void inspectSource()} disabled={busy}>检查数据源</button>

          {inspection && (
            <div className="inspection-card">
              <div><span>记录数</span><strong>{inspection.record_count?.toLocaleString() ?? "未知"}</strong></div>
              <div><span>字段</span><strong>{inspection.fields.length}</strong></div>
              <p className="mono wrap">{inspection.input_fingerprint}</p>
              <div className="field-chips">{inspection.fields.map((field) => <span key={field}>{field}</span>)}</div>
            </div>
          )}
        </div>

        <div className="setup-column">
          <div className="setup-step">
            <span className="step-number">2</span>
            <div><h3>字段映射与正文预览</h3><p>多个正文列按填写顺序用空行连接。</p></div>
          </div>
          <label>正文列（逗号分隔，有顺序）
            <input value={textFields} onChange={(event) => setTextFields(event.target.value)} placeholder="title, content" disabled={busy || !inspection} />
          </label>
          <div className="two-fields">
            <label>标题列
              <select value={titleField} onChange={(event) => setTitleField(event.target.value)} disabled={busy || !inspection}>
                <option value="">无</option>{inspection?.fields.map((field) => <option key={field}>{field}</option>)}
              </select>
            </label>
            <label>稳定 ID 列
              <select value={localIdField} onChange={(event) => setLocalIdField(event.target.value)} disabled={busy || !inspection}>
                <option value="">使用行位置</option>{inspection?.fields.map((field) => <option key={field}>{field}</option>)}
              </select>
            </label>
          </div>
          <div className="two-fields">
            <label>URL 列
              <select value={urlField} onChange={(event) => setUrlField(event.target.value)} disabled={busy || !inspection}>
                <option value="">无</option>{inspection?.fields.map((field) => <option key={field}>{field}</option>)}
              </select>
            </label>
            <label>Metadata 列（逗号）
              <input value={metadataFields} onChange={(event) => setMetadataFields(event.target.value)} placeholder="dataType" disabled={busy || !inspection} />
            </label>
          </div>
          <button className="secondary" onClick={() => void previewSource()} disabled={busy || !inspection}>生成 3 条预览</button>
          <div className="preview-stack">
            {preview.map((document, index) => (
              <article key={document.stable_locator}>
                <span>PREVIEW {index + 1} · {document.stable_locator}</span>
                <pre>{document.text}</pre>
              </article>
            ))}
          </div>
          <button className="primary" onClick={() => void importAndPrepare()} disabled={busy || !preview.length}>
            导入并生成 Review Snapshot
          </button>
          {(imported || prepared) && (
            <div className="snapshot-result">
              {imported && <p>Raw：<code>{imported.revision_directory}</code>{imported.reused_existing && "（已复用）"}</p>}
              {prepared && <p>Review：<code>{prepared.revision_directory}</code>{prepared.reused_existing && "（已复用）"}</p>}
            </div>
          )}
        </div>
      </div>

      <div className="catalog-section">
        <div className="setup-step">
          <span className="step-number">3</span>
          <div><h3>已有快照</h3><p>刷新网页后也能从这里继续，不会要求重新导入。</p></div>
        </div>
        {!catalog.length && <p className="muted">当前 label/data 中还没有完成的 Source Snapshot。</p>}
        <div className="catalog-list">
          {catalog.map((source) => (
            <div className="catalog-card" key={source.manifest.source_revision}>
              <div>
                <strong>{source.manifest.source_id}</strong>
                <span>{source.manifest.record_count.toLocaleString()} 条 · {source.manifest.license}</span>
                <code>{source.revision_directory}</code>
              </div>
              <div className="catalog-actions">
                {!source.prepares.length && (
                  <button onClick={() => void prepareExisting(source.revision_directory)} disabled={busy}>生成 Review</button>
                )}
                {source.prepares.map((value) => (
                  <button
                    key={value.manifest.prepare_revision}
                    className={prepareDirectory === value.revision_directory ? "selected" : ""}
                    onClick={() => selectPrepared(value)}
                    disabled={busy}
                  >使用 Review · {value.manifest.output_records.toLocaleString()} 条</button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="project-setup">
        <div className="setup-step">
          <span className="step-number">4</span>
          <div><h3>挂载项目并抽样</h3><p>项目固定引用当前 Review Snapshot；队列记录 seed，可重复生成。</p></div>
        </div>
        <div className="project-form">
          <label>已有项目
            <select value={projectId} onChange={(event) => setProjectId(event.target.value)} disabled={busy}>
              <option value="">创建新项目</option>
              {projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}
            </select>
          </label>
          {!projectId && <label>新项目名称<input value={newProjectName} onChange={(event) => setNewProjectName(event.target.value)} placeholder="中文预训练语料质量审核" disabled={busy} /></label>}
          <label>队列名称<input value={queueName} onChange={(event) => setQueueName(event.target.value)} disabled={busy} /></label>
          <label>抽样条数<input type="number" min={1} max={prepareRecords ?? undefined} value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} disabled={busy} /></label>
          <label>随机种子<input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} disabled={busy} /></label>
        </div>
        <div className="project-actions">
          <button className="primary" onClick={() => void attachAndCreateQueue()} disabled={busy || !prepareDirectory}>挂载所选 Review 并创建队列</button>
          {projectId && <button className="secondary" onClick={() => void createQueueForAttachedSources()} disabled={busy}>只为项目已有来源新建队列</button>}
        </div>
        {prepareDirectory && <p className="selected-snapshot">当前 Review：<code>{prepareDirectory}</code></p>}
      </div>
    </section>
  );
}
