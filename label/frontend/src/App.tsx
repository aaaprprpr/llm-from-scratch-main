import { useCallback, useEffect, useMemo, useState } from "react";
import { requestJson } from "./api";
import SetupWizard from "./SetupWizard";
import type {
  DocumentReview,
  Project,
  ProjectDetail,
  Queue,
  QueueDocument,
  ReviewBlock,
} from "./types";

const categories = [
  "encyclopedia",
  "news",
  "marketing",
  "fiction",
  "forum_or_social",
  "qa_or_instruction",
  "academic_or_technical",
  "code",
  "reference_or_table",
  "other",
];

const reviewFlags = [
  "advertisement",
  "boilerplate",
  "spam",
  "garbled",
  "duplicate",
  "low_information",
  "bad_format",
  "unsafe_or_pii",
  "wrong_language",
  "irrelevant",
  "other",
];

type TextView = "review" | "raw" | "materialized";

function queueSize(queue: Queue | null): number {
  if (!queue) return 0;
  return Object.values(queue.state_counts).reduce((sum, value) => sum + value, 0);
}

function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState("");
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [queueId, setQueueId] = useState("");
  const [ordinal, setOrdinal] = useState(0);
  const [document, setDocument] = useState<QueueDocument | null>(null);
  const [textView, setTextView] = useState<TextView>("review");
  const [quality, setQuality] = useState<number | null>(null);
  const [category, setCategory] = useState("");
  const [flags, setFlags] = useState<string[]>([]);
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("正在连接后端…");
  const [lastEventSeq, setLastEventSeq] = useState<number | null>(null);
  const [activeBlockOrdinal, setActiveBlockOrdinal] = useState(0);
  const [showSetup, setShowSetup] = useState(false);
  const [projectRefresh, setProjectRefresh] = useState(0);

  const selectedQueue = useMemo(
    () => project?.queues.find((queue) => queue.queue_id === queueId) ?? null,
    [project, queueId],
  );
  const totalItems = queueSize(selectedQueue);

  const loadProjects = useCallback(async () => {
    try {
      const values = await requestJson<Project[]>("/api/projects");
      setProjects(values);
      setProjectId((current) => current || values[0]?.project_id || "");
      if (!values.length) setShowSetup(true);
      setStatus(values.length ? "后端已连接" : "后端已连接；还没有项目");
    } catch (error) {
      setStatus(`无法连接后端：${String(error)}`);
    }
  }, []);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  useEffect(() => {
    if (!projectId) {
      setProject(null);
      return;
    }
    void requestJson<ProjectDetail>(`/api/projects/${projectId}`)
      .then((value) => {
        setProject(value);
        setQueueId((current) =>
          value.queues.some((queue) => queue.queue_id === current)
            ? current
            : value.queues[0]?.queue_id || "",
        );
      })
      .catch((error) => setStatus(`读取项目失败：${String(error)}`));
  }, [projectId, projectRefresh]);

  const finishSetup = useCallback(async (nextProjectId: string, nextQueueId: string) => {
    await loadProjects();
    setProjectId(nextProjectId);
    setQueueId(nextQueueId);
    setOrdinal(0);
    setProjectRefresh((value) => value + 1);
    setShowSetup(false);
  }, [loadProjects]);

  const loadDocument = useCallback(async () => {
    if (!queueId || totalItems === 0) {
      setDocument(null);
      return;
    }
    setBusy(true);
    try {
      const value = await requestJson<QueueDocument>(
        `/api/queues/${queueId}/items/${ordinal}`,
      );
      setDocument(value);
      setActiveBlockOrdinal(0);
      setProject((current) => current ? {
        ...current,
        queues: current.queues.map((queue) =>
          queue.queue_id === value.queue.queue_id ? value.queue : queue
        ),
      } : current);
      const review = value.document_review;
      setQuality(review?.quality ?? null);
      setCategory(review?.primary_category ?? "");
      setFlags(review?.flags ?? []);
      setNotes(review?.notes ?? "");
      setStatus(`已加载第 ${ordinal + 1} 条`);
    } catch (error) {
      setDocument(null);
      setStatus(`读取文档失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  }, [ordinal, queueId, totalItems]);

  useEffect(() => {
    void loadDocument();
  }, [loadDocument]);

  const go = useCallback(
    (next: number) => {
      if (!totalItems) return;
      setOrdinal(Math.max(0, Math.min(totalItems - 1, next)));
    },
    [totalItems],
  );

  const saveDocument = useCallback(
    async (decision: "keep" | "drop" | "unsure") => {
      if (!document) return;
      setBusy(true);
      setStatus("正在保存…");
      try {
        const result = await requestJson<{
          review: DocumentReview;
          event_seq: number;
        }>(`/api/reviews/documents/${document.document.doc_id}`, {
          method: "PUT",
          body: JSON.stringify({
            queue_id: queueId,
            ordinal,
            expected_revision: document.document_review?.revision ?? 0,
            decision,
            quality,
            primary_category: category || null,
            flags,
            notes,
          }),
        });
        setLastEventSeq(result.event_seq);
        setStatus(`已持久化：${decision}`);
        if (ordinal + 1 < totalItems) go(ordinal + 1);
        else await loadDocument();
      } catch (error) {
        setStatus(`保存失败：${String(error)}`);
        await loadDocument();
      } finally {
        setBusy(false);
      }
    }, [
      category,
      document,
      flags,
      go,
      loadDocument,
      notes,
      ordinal,
      quality,
      queueId,
      totalItems,
    ],
  );

  const saveBlock = useCallback(
    async (block: ReviewBlock, decision: "keep" | "drop") => {
      if (!document) return;
      setBusy(true);
      try {
        const result = await requestJson<{ event_seq: number }>(
          `/api/reviews/blocks/${block.block_id}`,
          {
            method: "PUT",
            body: JSON.stringify({
              queue_id: queueId,
              ordinal,
              expected_revision: block.review?.revision ?? 0,
              decision,
            }),
          },
        );
        setLastEventSeq(result.event_seq);
        setStatus(`Block ${block.ordinal + 1} 已设为 ${decision}`);
        await loadDocument();
      } catch (error) {
        setStatus(`Block 保存失败：${String(error)}`);
      } finally {
        setBusy(false);
      }
    },
    [document, loadDocument, ordinal, queueId],
  );

  const undo = useCallback(async () => {
    if (!lastEventSeq || !projectId) return;
    setBusy(true);
    try {
      await requestJson(`/api/events/${lastEventSeq}/undo`, {
        method: "POST",
        body: JSON.stringify({ project_id: projectId }),
      });
      setLastEventSeq(null);
      setStatus("已撤销最近一次操作");
      await loadDocument();
    } catch (error) {
      setStatus(`撤销失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  }, [lastEventSeq, loadDocument, projectId]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (event.ctrlKey && event.key.toLowerCase() === "z") {
        event.preventDefault();
        void undo();
        return;
      }
      if (typing || busy) return;
      if (event.key.toLowerCase() === "a") void saveDocument("keep");
      if (event.key.toLowerCase() === "d") void saveDocument("drop");
      if (event.key.toLowerCase() === "s") void saveDocument("unsure");
      if (/^[0-3]$/.test(event.key)) setQuality(Number(event.key));
      if (event.key.toLowerCase() === "x" && document?.blocks[activeBlockOrdinal]) {
        void saveBlock(document.blocks[activeBlockOrdinal], "drop");
      }
      if (event.key.toLowerCase() === "j" || event.key === "ArrowRight") {
        go(ordinal + 1);
      }
      if (event.key.toLowerCase() === "k" || event.key === "ArrowLeft") {
        go(ordinal - 1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeBlockOrdinal, busy, document, go, ordinal, saveBlock, saveDocument, undo]);

  const activeText = document
    ? textView === "raw"
      ? document.raw_text
      : textView === "materialized"
        ? document.materialized_text
        : document.review_text
    : "";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">LOCAL CURATION / V0.1</p>
          <h1>语料审核台</h1>
        </div>
        <div className="topbar-actions">
          <button className="ghost" onClick={() => setShowSetup((value) => !value)}>
            {showSetup ? "返回审核" : "导入 / 建队列"}
          </button>
          <div className={`status ${status.includes("失败") ? "error" : ""}`}>
            <span className="status-dot" />
            {status}
          </div>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar panel">
          <label>
            项目
            <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
              <option value="">选择项目</option>
              {projects.map((value) => (
                <option key={value.project_id} value={value.project_id}>
                  {value.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            审核队列
            <select
              value={queueId}
              onChange={(e) => {
                setQueueId(e.target.value);
                setOrdinal(0);
              }}
            >
              <option value="">选择队列</option>
              {project?.queues.map((queue) => (
                <option key={queue.queue_id} value={queue.queue_id}>
                  {queue.name}
                </option>
              ))}
            </select>
          </label>

          {selectedQueue && (
            <>
              <div className="progress-copy">
                <span>队列进度</span>
                <strong>
                  {selectedQueue.state_counts.done ?? 0} / {totalItems}
                </strong>
              </div>
              <div className="progress-track">
                <span
                  style={{
                    width: `${totalItems ? ((selectedQueue.state_counts.done ?? 0) / totalItems) * 100 : 0}%`,
                  }}
                />
              </div>
              <dl className="compact-stats">
                <div><dt>Pending</dt><dd>{selectedQueue.state_counts.pending ?? 0}</dd></div>
                <div><dt>Done</dt><dd>{selectedQueue.state_counts.done ?? 0}</dd></div>
                <div><dt>Seed</dt><dd>{selectedQueue.sampling_seed}</dd></div>
                {Object.entries(selectedQueue.source_counts).map(([source, count]) => (
                  <div key={source}><dt>{source}</dt><dd>{count}</dd></div>
                ))}
              </dl>
            </>
          )}

          <div className="pager">
            <button onClick={() => go(ordinal - 1)} disabled={ordinal <= 0}>←</button>
            <label>
              <input
                type="number"
                min={1}
                max={Math.max(1, totalItems)}
                value={ordinal + 1}
                onChange={(e) => go(Number(e.target.value) - 1)}
              />
              <span>/ {totalItems || "—"}</span>
            </label>
            <button onClick={() => go(ordinal + 1)} disabled={ordinal + 1 >= totalItems}>→</button>
          </div>
          <button className="ghost full" onClick={() => void undo()} disabled={!lastEventSeq || busy}>
            撤销最近操作 <kbd>Ctrl Z</kbd>
          </button>
          <p className="shortcut-help">
            A 保留 · D 丢弃 · S 待定<br />
            0–3 质量 · X 丢弃当前 Block<br />
            J/K 或方向键切换
          </p>
        </aside>

        <main className="document-panel panel">
          {showSetup ? (
            <SetupWizard
              projects={projects}
              initialProjectId={projectId}
              onComplete={finishSetup}
              onClose={() => setShowSetup(false)}
            />
          ) : !document ? (
            <div className="empty-state">
              <span>⌁</span>
              <h2>选择一个已有审核队列</h2>
              <p>当前还没有可浏览的队列，可以从本地 HF Dataset、JSONL 或 TXT 开始。</p>
              <button className="primary" onClick={() => setShowSetup(true)}>导入数据集</button>
            </div>
          ) : (
            <>
              <div className="document-heading">
                <div>
                  <p className="mono">DOC {document.document.doc_id}</p>
                  <h2>{document.provenance.title || "未命名文档"}</h2>
                </div>
                <div className={`decision-badge ${document.document_review?.decision ?? "unreviewed"}`}>
                  {document.document_review?.decision ?? "unreviewed"}
                </div>
              </div>

              <div className="review-controls">
                <div className="decision-buttons">
                  <button className="keep" onClick={() => void saveDocument("keep")} disabled={busy}>A · 保留</button>
                  <button className="drop" onClick={() => void saveDocument("drop")} disabled={busy}>D · 丢弃</button>
                  <button className="unsure" onClick={() => void saveDocument("unsure")} disabled={busy}>S · 待定</button>
                </div>
                <div className="quality-row">
                  <span>质量</span>
                  {[0, 1, 2, 3].map((value) => (
                    <button
                      key={value}
                      className={quality === value ? "active" : ""}
                      onClick={() => setQuality(value)}
                    >{value}</button>
                  ))}
                  <select value={category} onChange={(e) => setCategory(e.target.value)}>
                    <option value="">内容类型</option>
                    {categories.map((value) => <option key={value}>{value}</option>)}
                  </select>
                </div>
                <details className="flags-control">
                  <summary>缺陷 Flags {flags.length ? `(${flags.length})` : ""}</summary>
                  <div className="flag-grid">
                    {reviewFlags.map((flag) => (
                      <label key={flag}>
                        <input
                          type="checkbox"
                          checked={flags.includes(flag)}
                          onChange={() => setFlags((current) =>
                            current.includes(flag)
                              ? current.filter((value) => value !== flag)
                              : [...current, flag]
                          )}
                        />
                        {flag}
                      </label>
                    ))}
                  </div>
                </details>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="审核备注（会随下一次决定一起保存）"
                  rows={2}
                />
              </div>

              <div className="text-tabs">
                {(["raw", "review", "materialized"] as TextView[]).map((view) => (
                  <button
                    key={view}
                    className={textView === view ? "active" : ""}
                    onClick={() => setTextView(view)}
                  >{view}</button>
                ))}
                <span>
                  {Array.from(activeText).length.toLocaleString()} 字符
                  {document.token_counts &&
                    ` · ${document.token_counts[textView].toLocaleString()} tokens`}
                </span>
              </div>
              <article className="text-view">{activeText}</article>

              <section className="blocks-section">
                <div className="section-heading">
                  <h3>Blocks</h3>
                  <span>{document.blocks.length} 个确定性段落</span>
                </div>
                {document.blocks.map((block) => (
                  <div
                    className={`block-card ${block.review?.decision === "drop" ? "is-dropped" : ""} ${activeBlockOrdinal === block.ordinal ? "is-active" : ""}`}
                    key={block.block_id}
                    onMouseEnter={() => setActiveBlockOrdinal(block.ordinal)}
                    onClick={() => setActiveBlockOrdinal(block.ordinal)}
                  >
                    <div className="block-meta">
                      <span>#{block.ordinal + 1}</span>
                      <span>cp {block.start_cp}–{block.end_cp}</span>
                      <div>
                        <button onClick={() => void saveBlock(block, "keep")} disabled={busy}>保留</button>
                        <button onClick={() => void saveBlock(block, "drop")} disabled={busy}>丢弃</button>
                      </div>
                    </div>
                    <p>{block.text}</p>
                  </div>
                ))}
              </section>
            </>
          )}
        </main>

        <aside className="provenance panel">
          <p className="eyebrow">PROVENANCE</p>
          {document ? (
            <dl>
              <div><dt>Source</dt><dd>{document.provenance.source_id}</dd></div>
              <div><dt>Source row</dt><dd>{document.provenance.source_row.toLocaleString()}</dd></div>
              <div><dt>Local ID</dt><dd>{document.provenance.source_local_id || "—"}</dd></div>
              <div><dt>License</dt><dd>{document.provenance.license}</dd></div>
              <div><dt>Revision</dt><dd className="mono wrap">{document.provenance.source_revision}</dd></div>
              <div><dt>Original</dt><dd className="wrap">{document.provenance.original_location}</dd></div>
              {document.provenance.url && (
                <div><dt>URL</dt><dd><a href={document.provenance.url} target="_blank" rel="noreferrer">打开来源 ↗</a></dd></div>
              )}
            </dl>
          ) : <p className="muted">选择文档后显示不可变来源链。</p>}
        </aside>
      </div>
    </div>
  );
}

export default App;
