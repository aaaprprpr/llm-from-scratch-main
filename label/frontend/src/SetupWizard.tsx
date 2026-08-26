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
  manifest: ImportResult["source_manifest"] & { original_location: string };
  prepares: CatalogPrepare[];
};

type Props = {
  projects: Project[];
  initialProjectId: string;
  onComplete: (projectId: string, queueId: string) => Promise<void> | void;
};

type Mapping = {
  text_fields: string[];
  text_separator: string;
  title_field: string | null;
  url_field: string | null;
  local_id_field: string | null;
  metadata_fields: string[];
};

function commaValues(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function findField(fields: string[], candidates: string[]): string | null {
  const byLowercase = new Map(fields.map((field) => [field.toLowerCase(), field]));
  for (const candidate of candidates) {
    const field = byLowercase.get(candidate.toLowerCase());
    if (field) return field;
  }
  return null;
}

function inferMapping(fields: string[]): Mapping {
  const title = findField(fields, ["title", "name"]);
  const directText = findField(fields, ["text"]);
  const content = findField(fields, ["content", "body", "article", "document"]);
  const textFields = directText
    ? [directText]
    : content
      ? title && title !== content ? [title, content] : [content]
      : fields.length === 1 ? [fields[0]] : [];
  const localId = findField(fields, ["uniqueKey", "doc_id", "document_id", "id"]);
  const url = findField(fields, ["url", "link", "source_url"]);
  const dataType = findField(fields, ["dataType", "category", "source"]);
  return {
    text_fields: textFields,
    text_separator: "\n\n",
    title_field: title,
    url_field: url,
    local_id_field: localId,
    metadata_fields: dataType ? [dataType] : [],
  };
}

function pathName(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) || path;
}

export default function SetupWizard({ projects, initialProjectId, onComplete }: Props) {
  const [adapter, setAdapter] = useState("huggingface_local");
  const [path, setPath] = useState("");
  const [optionsJson, setOptionsJson] = useState("{}");
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [textFields, setTextFields] = useState("");
  const [titleField, setTitleField] = useState("");
  const [urlField, setUrlField] = useState("");
  const [localIdField, setLocalIdField] = useState("");
  const [metadataFields, setMetadataFields] = useState("");
  const [imported, setImported] = useState<ImportResult | null>(null);
  const [prepared, setPrepared] = useState<PrepareResult | null>(null);
  const [prepareDirectory, setPrepareDirectory] = useState("");
  const [prepareRecords, setPrepareRecords] = useState<number | null>(null);
  const [selectedDatasetName, setSelectedDatasetName] = useState("");
  const [catalog, setCatalog] = useState<CatalogSource[]>([]);
  const [projectId, setProjectId] = useState(initialProjectId);
  const [newProjectName, setNewProjectName] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("选择本机上的数据集目录。不会通过浏览器上传数据。");

  const mapping = useMemo<Mapping>(() => ({
    text_fields: commaValues(textFields),
    text_separator: "\n\n",
    title_field: titleField || null,
    url_field: urlField || null,
    local_id_field: localIdField || null,
    metadata_fields: commaValues(metadataFields),
  }), [localIdField, metadataFields, textFields, titleField, urlField]);

  const loadCatalog = async () => {
    setCatalog(await requestJson<CatalogSource[]>("/api/catalog"));
  };

  useEffect(() => {
    void loadCatalog().catch((error) => setMessage(`读取已有数据失败：${String(error)}`));
  }, []);

  useEffect(() => {
    if (initialProjectId) setProjectId(initialProjectId);
  }, [initialProjectId]);

  const sourceOptions = () => {
    const value = JSON.parse(optionsJson || "{}");
    if (!value || Array.isArray(value) || typeof value !== "object") {
      throw new Error("读取选项必须是 JSON object");
    }
    return value as Record<string, unknown>;
  };

  const applyAutomaticMapping = (fields: string[]) => {
    const inferred = inferMapping(fields);
    setTextFields(inferred.text_fields.join(", "));
    setTitleField(inferred.title_field ?? "");
    setUrlField(inferred.url_field ?? "");
    setLocalIdField(inferred.local_id_field ?? "");
    setMetadataFields(inferred.metadata_fields.join(", "));
    return inferred;
  };

  const inspectSource = async (selectedPath = path) => {
    if (!selectedPath.trim()) {
      setMessage("还没有选择数据集。");
      return;
    }
    setBusy(true);
    setMessage("正在识别数据集…");
    try {
      const result = await requestJson<Inspection>("/api/imports/inspect", {
        method: "POST",
        body: JSON.stringify({ adapter, path: selectedPath, options: sourceOptions() }),
      });
      setPath(result.input_location || selectedPath);
      setSelectedDatasetName(pathName(result.input_location || selectedPath));
      setInspection(result);
      setImported(null);
      setPrepared(null);
      const inferred = applyAutomaticMapping(result.fields);
      setMessage(
        inferred.text_fields.length
          ? `已识别 ${result.record_count?.toLocaleString() ?? "未知"} 条记录，可以直接导入。`
          : "没有自动找到正文列，请展开“识别有误时修改”。",
      );
    } catch (error) {
      setInspection(null);
      setMessage(`识别失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const choosePath = async () => {
    setBusy(true);
    setMessage("正在打开本机选择窗口…");
    try {
      const result = await requestJson<{ path: string | null }>("/api/system/select-path", {
        method: "POST",
        body: JSON.stringify({
          kind: adapter === "huggingface_local" ? "directory" : "file",
          adapter,
          initial_directory: path || null,
        }),
      });
      if (!result.path) {
        setMessage("没有选择数据集。");
        return;
      }
      setPath(result.path);
      await inspectSource(result.path);
    } catch (error) {
      setMessage(`选择失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const importAndPrepare = async () => {
    if (!inspection || !mapping.text_fields.length) {
      setMessage("数据集还没有正确识别。");
      return;
    }
    setBusy(true);
    try {
      setMessage("正在导入数据；进度显示在后端终端中…");
      const importResult = await requestJson<ImportResult>("/api/imports", {
        method: "POST",
        body: JSON.stringify({ adapter, path, options: sourceOptions(), mapping, max_shard_size: "1GB" }),
      });
      setImported(importResult);
      setMessage("导入完成，正在准备清洗数据…");
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
      setSelectedDatasetName(pathName(inspection.input_location));
      setPrepareDirectory(prepareResult.revision_directory);
      setPrepareRecords(prepareResult.manifest.output_records);
      await loadCatalog();
      setMessage(`数据已就绪，共 ${prepareResult.manifest.output_records.toLocaleString()} 条。`);
    } catch (error) {
      setMessage(`导入失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const prepareExisting = async (revisionDirectory: string, originalLocation: string) => {
    setBusy(true);
    setMessage("正在准备已有数据…");
    try {
      const result = await requestJson<PrepareResult>("/api/prepares", {
        method: "POST",
        body: JSON.stringify({ source_revision_directory: revisionDirectory, config: {} }),
      });
      setPrepared(result);
      setSelectedDatasetName(pathName(originalLocation));
      setPrepareDirectory(result.revision_directory);
      setPrepareRecords(result.manifest.output_records);
      await loadCatalog();
      setMessage("数据已就绪，可以开始全量清洗。");
    } catch (error) {
      setMessage(`准备失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const selectPrepared = (value: CatalogPrepare, originalLocation: string) => {
    setPrepareDirectory(value.revision_directory);
    setPrepareRecords(value.manifest.output_records);
    setSelectedDatasetName(pathName(originalLocation));
    setMessage(`已选择 ${value.manifest.output_records.toLocaleString()} 条清洗数据。`);
  };

  const ensureProject = async (): Promise<string> => {
    if (projectId) return projectId;
    const projectName = newProjectName.trim() || `${selectedDatasetName || "语料"}清洗`;
    const project = await requestJson<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name: projectName }),
    });
    setProjectId(project.project_id);
    return project.project_id;
  };

  const createQueue = async (targetProjectId: string) => requestJson<{ queue_id: string }>(
    `/api/projects/${targetProjectId}/queues`,
    {
      method: "POST",
      body: JSON.stringify({
        name: "全量人工清洗",
        policy: "full_dataset",
      }),
    },
  );

  const attachAndCreateQueue = async () => {
    if (!prepareDirectory) {
      setMessage("请先导入或选择一份已有数据。");
      return;
    }
    setBusy(true);
    try {
      const targetProjectId = await ensureProject();
      setMessage("正在开始全量清洗…");
      await requestJson(`/api/projects/${targetProjectId}/sources`, {
        method: "POST",
        body: JSON.stringify({ prepare_revision_directory: prepareDirectory }),
      });
      const queue = await createQueue(targetProjectId);
      await onComplete(targetProjectId, queue.queue_id);
    } catch (error) {
      setMessage(`开始清洗失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const changeAdapter = (value: string) => {
    setAdapter(value);
    setPath("");
    setInspection(null);
    setImported(null);
    setPrepared(null);
    setMessage(value === "huggingface_local" ? "选择数据集目录。" : "选择一个数据文件。");
  };

  return (
    <section className="setup-wizard">
      <div className="setup-title">
        <h2>导入数据</h2>
        <p>把数据登记到工具后即可逐条全量清洗；不会修改原始目录。</p>
      </div>

      <div className={`setup-message ${message.includes("失败") ? "error" : ""}`}>
        {busy && <span className="spinner" />}
        {message}
      </div>

      <section className="source-card">
        <div className="source-picker-row">
          <label>数据格式
            <select value={adapter} onChange={(event) => changeAdapter(event.target.value)} disabled={busy}>
              <option value="huggingface_local">本地数据集目录</option>
              <option value="jsonl">逐行 JSON 文件</option>
              <option value="text">纯文本文件</option>
            </select>
          </label>
          <div className={`chosen-path ${path ? "has-value" : ""}`}>
            <span>{path ? pathName(path) : "尚未选择数据"}</span>
            {path && <small title={path}>{path}</small>}
          </div>
          <button className="primary choose-path" onClick={() => void choosePath()} disabled={busy}>
            {adapter === "huggingface_local" ? "选择文件夹" : "选择文件"}
          </button>
        </div>

        {inspection && (
          <div className="dataset-summary">
            <div><span>数据量</span><strong>{inspection.record_count?.toLocaleString() ?? "未知"} 条</strong></div>
            <div><span>识别正文</span><strong>{mapping.text_fields.join(" + ") || "未识别"}</strong></div>
            <div><span>稳定标识</span><strong>{mapping.local_id_field || "行号"}</strong></div>
            <button className="primary import-button" onClick={() => void importAndPrepare()} disabled={busy || !mapping.text_fields.length}>
              导入数据集
            </button>
          </div>
        )}
        {inspection && <p className="source-help">导入会在 label/data 中建立清洗用副本，原目录保持不变；大数据集会额外占用磁盘。</p>}

        <details className="recognition-settings" open={Boolean(inspection && !mapping.text_fields.length)}>
          <summary>识别有误时修改</summary>
          <div className="recognition-grid">
            <label>正文列<input value={textFields} onChange={(event) => setTextFields(event.target.value)} placeholder="text 或 title, content" disabled={busy || !inspection} /></label>
            <label>标题列
              <select value={titleField} onChange={(event) => setTitleField(event.target.value)} disabled={busy || !inspection}>
                <option value="">无</option>{inspection?.fields.map((field) => <option key={field}>{field}</option>)}
              </select>
            </label>
            <label>唯一标识列
              <select value={localIdField} onChange={(event) => setLocalIdField(event.target.value)} disabled={busy || !inspection}>
                <option value="">使用行号</option>{inspection?.fields.map((field) => <option key={field}>{field}</option>)}
              </select>
            </label>
            <label>网页地址列
              <select value={urlField} onChange={(event) => setUrlField(event.target.value)} disabled={busy || !inspection}>
                <option value="">无</option>{inspection?.fields.map((field) => <option key={field}>{field}</option>)}
              </select>
            </label>
            <label>附加字段<input value={metadataFields} onChange={(event) => setMetadataFields(event.target.value)} disabled={busy || !inspection} /></label>
          </div>
          <p>可用字段：{inspection?.fields.join("、") || "选择数据后显示"}</p>
        </details>

        <details className="manual-fallback">
          <summary>无法打开选择窗口</summary>
          <div className="manual-path-row">
            <input value={path} onChange={(event) => setPath(event.target.value)} placeholder="输入运行后端这台机器上的路径" disabled={busy} />
            <button className="secondary" onClick={() => void inspectSource()} disabled={busy || !path}>读取路径</button>
          </div>
          {adapter !== "huggingface_local" && (
            <textarea value={optionsJson} onChange={(event) => setOptionsJson(event.target.value)} rows={2} disabled={busy} aria-label="读取选项" />
          )}
        </details>

        {(imported || prepared) && (
          <p className="snapshot-result">
            {prepared ? `已准备 ${prepared.manifest.output_records.toLocaleString()} 条清洗数据` : "原始数据已导入"}
            {(imported?.reused_existing || prepared?.reused_existing) && "（复用已有结果）"}
          </p>
        )}
      </section>

      {catalog.length > 0 && (
        <details className="catalog-section">
          <summary>切换到以前导入的数据（{catalog.length}）</summary>
          <p className="catalog-help">只有需要换数据时才用这里，可以避免重复导入和复制。</p>
          <div className="catalog-list">
            {catalog.map((source) => (
              <div className="catalog-card" key={source.manifest.source_revision}>
                <div>
                  <strong>{pathName(source.manifest.original_location)}</strong>
                  <span>{source.manifest.record_count.toLocaleString()} 条</span>
                  <small title={source.manifest.original_location}>{source.manifest.original_location}</small>
                </div>
                <div className="catalog-actions">
                  {!source.prepares.length && (
                    <button onClick={() => void prepareExisting(source.revision_directory, source.manifest.original_location)} disabled={busy}>准备清洗数据</button>
                  )}
                  {source.prepares.map((value) => (
                    <button
                      key={value.manifest.prepare_revision}
                      className={prepareDirectory === value.revision_directory ? "selected" : ""}
                      onClick={() => selectPrepared(value, source.manifest.original_location)}
                      disabled={busy}
                    >选择 · {value.manifest.output_records.toLocaleString()} 条</button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </details>
      )}

      <section className="project-setup">
        <div className="section-title">
          <h3>开始全量人工清洗</h3>
          <span>{prepareRecords ? `${selectedDatasetName || "当前数据"} · 共 ${prepareRecords.toLocaleString()} 条` : "先选择或导入数据"}</span>
        </div>
        <div className="full-cleaning-summary">
          <strong>{prepareRecords?.toLocaleString() ?? "—"} 条全部进入清洗流程</strong>
          <p>
            按数据集顺序逐条处理，没有抽样，也不会遗漏文档。你可以整篇保留或删除，也可以删除文档中的低质量段落。
            清洗记录将保存到{projectId
              ? `已有项目“${projects.find((project) => project.project_id === projectId)?.name ?? projectId}”中`
              : "自动创建的新项目中"}。
          </p>
        </div>
        <details className="queue-advanced">
          <summary>高级设置</summary>
          <div className="project-form">
            <label>归入项目
              <select value={projectId} onChange={(event) => setProjectId(event.target.value)} disabled={busy}>
                <option value="">自动新建项目</option>
                {projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}
              </select>
            </label>
            {!projectId && <label>项目名称<input value={newProjectName} onChange={(event) => setNewProjectName(event.target.value)} placeholder={`${selectedDatasetName || "语料"}清洗`} disabled={busy} /></label>}
          </div>
        </details>
        <div className="project-actions">
          <button className="primary" onClick={() => void attachAndCreateQueue()} disabled={busy || !prepareDirectory}>开始全量清洗</button>
        </div>
      </section>
    </section>
  );
}
