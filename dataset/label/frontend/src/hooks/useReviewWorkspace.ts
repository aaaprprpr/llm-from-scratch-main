import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { requestJson } from "../api";
import {
  renderEditableBlocks,
  serializeEditableBlocks,
  type EditAction,
  type EditableBlock,
} from "../BlockEditor";
import {
  cloneBlocks,
  decisionLabels,
  makeEditableBlocks,
  queueSize,
} from "../reviewState";
import type { LlmCleaningResult, Project, ProjectDetail, QueueDocument } from "../types";

export function useReviewWorkspace(showSettings = false) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState("");
  const [chosenQueueId, setChosenQueueId] = useState("");
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [ordinal, setOrdinal] = useState(0);
  const [pageInput, setPageInput] = useState("1");
  const [document, setDocument] = useState<QueueDocument | null>(null);
  const [draftBlocks, setDraftBlocks] = useState<EditableBlock[]>([]);
  const [editorText, setEditorText] = useState("");
  const [textDirty, setTextDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [llmCleaning, setLlmCleaning] = useState(false);
  const [llmResult, setLlmResult] = useState<LlmCleaningResult | null>(null);
  const llmInFlight = useRef(false);
  const documentVersion = useRef(0);
  const lastBatchRefresh = useRef("");
  const [status, setStatus] = useState("正在连接后端…");
  const [activeBlockId, setActiveBlockId] = useState<string | null>(null);
  const undoStack = useRef<EditableBlock[][]>([]);
  const redoStack = useRef<EditableBlock[][]>([]);
  const typingGroup = useRef<{ blockId: string; at: number } | null>(null);
  const [showSetup, setShowSetup] = useState(false);
  const [projectRefresh, setProjectRefresh] = useState(0);
  const [cleanProgress, setCleanProgress] = useState<{ completed: number; processed: number; attempts: number;
    manual_completed: number; manual_llm_saved: number;
    batch_counts: { keep?: number; drop?: number; incomplete?: number } } | null>(null);

  const selectedQueue = useMemo(() => {
    const fullQueues = project?.queues.filter(
      (queue) => queue.sampling_policy.type === "full_dataset",
    ) ?? [];
    return fullQueues.find((queue) => queue.queue_id === chosenQueueId)
      ?? fullQueues[fullQueues.length - 1] ?? null;
  }, [project, chosenQueueId]);
  const queueId = selectedQueue?.queue_id ?? "";
  const totalItems = queueSize(selectedQueue);
  const batchDecision = document?.batch_clean?.status;
  const activeDecision = document?.effective_source === "batch" && (batchDecision === "keep" || batchDecision === "drop")
    ? batchDecision : document?.document_review?.decision ?? "unreviewed";
  const activeDecisionOrigin = document?.effective_source === "batch" ? "批量 LLM · " : "";

  const refreshCleanProgress = useCallback(async () => {
    if (!queueId) { setCleanProgress(null); return; }
    try {
      const value = await requestJson<{ completed: number; processed: number; attempts: number;
        manual_completed: number; manual_llm_saved: number;
        batch_counts: { keep?: number; drop?: number; incomplete?: number } }>(
        `/api/auto-clean/${queueId}`,
      );
      setCleanProgress(value);
    } catch { /* Keep the last visible progress during a transient refresh error. */ }
  }, [queueId]);

  useEffect(() => {
    void refreshCleanProgress();
    if (!queueId) return;
    const timer = window.setInterval(() => void refreshCleanProgress(), 2000);
    return () => window.clearInterval(timer);
  }, [queueId, refreshCleanProgress]);

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
    setProject(null);
    if (!projectId) return;
    let active = true;
    void requestJson<ProjectDetail>(`/api/projects/${projectId}`)
      .then((value) => { if (active) setProject(value); })
      .catch((error) => { if (active) setStatus(`读取项目失败：${String(error)}`); });
    return () => { active = false; };
  }, [projectId, projectRefresh]);

  const finishSetup = useCallback(async (nextProjectId: string, nextQueueId: string) => {
    await loadProjects();
    setProjectId(nextProjectId);
    setChosenQueueId(nextQueueId);
    setOrdinal(0);
    setProjectRefresh((value) => value + 1);
    setShowSetup(false);
  }, [loadProjects]);

  const loadDocument = useCallback(async () => {
    const version = ++documentVersion.current;
    setLlmResult(null);
    if (!queueId || totalItems === 0) {
      setDocument(null);
      setBusy(false);
      return;
    }
    setBusy(true);
    try {
      const value = await requestJson<QueueDocument>(
        `/api/queues/${queueId}/items/${ordinal}`,
      );
      if (version !== documentVersion.current) return;
      setDocument(value);
      const nextBlocks = makeEditableBlocks(value);
      setDraftBlocks(nextBlocks);
      setEditorText(value.materialized_text);
      setTextDirty(false);
      setActiveBlockId(nextBlocks[0]?.id ?? null);
      undoStack.current = [];
      redoStack.current = [];
      typingGroup.current = null;
      setProject((current) => current ? {
        ...current,
        queues: current.queues.map((queue) =>
          queue.queue_id === value.queue.queue_id ? value.queue : queue
        ),
      } : current);
      setStatus(`已加载第 ${ordinal + 1} 条`);
    } catch (error) {
      if (version !== documentVersion.current) return;
      setDocument(null);
      setStatus(`读取文档失败：${String(error)}`);
    } finally {
      if (version === documentVersion.current) setBusy(false);
    }
  }, [ordinal, queueId, totalItems]);

  useEffect(() => {
    void loadDocument();
  }, [loadDocument]);

  useEffect(() => {
    setPageInput(String(ordinal + 1));
  }, [ordinal]);

  useEffect(() => {
    if (!document || busy || textDirty || document.item.ordinal !== ordinal) return;
    if (document.document_review?.decision === "keep" || document.document_review?.decision === "drop") return;
    if (document.document_review?.edited_text != null) return;
    const hasNewResult = document.batch_clean === null && (cleanProgress?.attempts ?? 0) > 0;
    const mayHaveRetried = document.batch_clean?.status === "incomplete";
    const refreshKey = `${queueId}:${ordinal}:${cleanProgress?.attempts ?? 0}`;
    if ((hasNewResult || mayHaveRetried) && lastBatchRefresh.current !== refreshKey) {
      lastBatchRefresh.current = refreshKey;
      void loadDocument();
    }
  }, [busy, cleanProgress?.attempts, cleanProgress?.processed, document, loadDocument,
      ordinal, queueId, textDirty]);

  useEffect(() => {
    if (!queueId || totalItems <= 0) return;
    const saved = Number(window.localStorage.getItem(`label.queue.${queueId}.ordinal`));
    if (Number.isInteger(saved) && saved >= 0) {
      setOrdinal(Math.min(totalItems - 1, saved));
    }
  }, [queueId, totalItems]);

  const go = useCallback((next: number) => {
    if (!totalItems) return;
    const bounded = Math.max(0, Math.min(totalItems - 1, next));
    setOrdinal(bounded);
    if (queueId) {
      window.localStorage.setItem(`label.queue.${queueId}.ordinal`, String(bounded));
    }
  }, [queueId, totalItems]);

  const saveDocument = useCallback(async (
    decision: "keep" | "drop" | "unsure",
    destination: number = ordinal + 1,
  ) => {
    if (!document || busy || llmInFlight.current) return;
    if (decision !== "drop" && !editorText.trim()) {
      setStatus("清洗结果为空；如果整篇不要，请选择丢弃");
      return;
    }
    setBusy(true);
    setStatus("正在保存…");
    try {
      await requestJson(`/api/reviews/documents/${document.document.doc_id}`, {
        method: "PUT",
        body: JSON.stringify({
          queue_id: queueId,
          ordinal,
          expected_revision: document.document_review?.revision ?? 0,
          decision,
          quality: document.document_review?.quality ?? null,
          primary_category: document.document_review?.primary_category ?? null,
          flags: document.document_review?.flags ?? [],
          notes: document.document_review?.notes ?? "",
          edited_text: decision === "drop" ? null : editorText,
        }),
      });
      setTextDirty(false);
      void refreshCleanProgress();
      setStatus(`已保存：${decisionLabels[decision]}`);
      if (destination >= 0 && destination < totalItems && destination !== ordinal) {
        go(destination);
      } else {
        await loadDocument();
      }
    } catch (error) {
      setStatus(`保存失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  }, [
    busy,
    document,
    editorText,
    go,
    loadDocument,
    ordinal,
    queueId,
    refreshCleanProgress,
    totalItems,
  ]);

  const applyDraftBlocks = useCallback((nextBlocks: EditableBlock[]) => {
    setDraftBlocks(nextBlocks);
    const nextText = renderEditableBlocks(nextBlocks);
    setEditorText(nextText);
    setTextDirty(nextText !== document?.materialized_text);
  }, [document?.materialized_text]);

  const updateDraftBlocks = useCallback((
    nextBlocks: EditableBlock[],
    action: EditAction = { kind: "command" },
  ) => {
    setLlmResult(null);
    const now = window.performance.now();
    const groupedTyping = action.kind === "typing"
      && typingGroup.current?.blockId === action.blockId
      && now - typingGroup.current.at < 800;
    if (!groupedTyping) {
      undoStack.current.push(cloneBlocks(draftBlocks));
      if (undoStack.current.length > 200) undoStack.current.shift();
    }
    redoStack.current = [];
    typingGroup.current = action.kind === "typing"
      ? { blockId: action.blockId, at: now }
      : null;
    applyDraftBlocks(nextBlocks);
  }, [applyDraftBlocks, draftBlocks]);

  const cleanCurrentDocument = useCallback(async () => {
    if (!document || busy || llmInFlight.current) return;
    const blocks = serializeEditableBlocks(draftBlocks);
    if (!blocks.some((block) => block.text.trim())) {
      setStatus("当前正文为空，没有可清洗的内容");
      return;
    }
    const version = documentVersion.current;
    llmInFlight.current = true;
    setBusy(true);
    setLlmCleaning(true);
    setStatus("LLM 正在检查本条全文，长文会分块处理，请稍候…");
    try {
      const result = await requestJson<LlmCleaningResult>(
        `/api/reviews/documents/${document.document.doc_id}/llm-clean`,
        {
          method: "POST",
          body: JSON.stringify({
            queue_id: queueId, ordinal,
            expected_revision: document.document_review?.revision ?? 0,
            content_sha256: document.document.content_sha256,
            blocks,
          }),
        },
      );
      if (version !== documentVersion.current) return;
      if (result.saved) {
        void refreshCleanProgress();
        await loadDocument();
        setLlmResult(result);
        setStatus(`已清洗并保存第 ${ordinal + 1} 条：删除 ${result.removals.length} 处、修订 ${result.edits.length} 处`);
      } else {
        setLlmResult(result);
        setStatus(`第 ${ordinal + 1} 条有未完成分块，未写入清洗结果；再次点击会重试未完成分块`);
      }
    } catch (error) {
      if (version === documentVersion.current) setStatus(`LLM 清洗失败：${String(error)}；草稿已保留`);
    } finally {
      llmInFlight.current = false;
      setLlmCleaning(false);
      if (version === documentVersion.current) setBusy(false);
    }
  }, [busy, document, draftBlocks, ordinal, queueId, loadDocument, refreshCleanProgress]);

  const undoDraft = useCallback(() => {
    if (busy || llmInFlight.current) return;
    const previous = undoStack.current.pop();
    if (!previous) return;
    redoStack.current.push(cloneBlocks(draftBlocks));
    typingGroup.current = null;
    applyDraftBlocks(previous);
    setLlmResult(null);
    setStatus("已撤销本次文档修改");
  }, [applyDraftBlocks, busy, draftBlocks]);

  const redoDraft = useCallback(() => {
    if (busy || llmInFlight.current) return;
    const next = redoStack.current.pop();
    if (!next) return;
    undoStack.current.push(cloneBlocks(draftBlocks));
    typingGroup.current = null;
    applyDraftBlocks(next);
    setLlmResult(null);
    setStatus("已重做本次文档修改");
  }, [applyDraftBlocks, busy, draftBlocks]);

  const saveAndGo = useCallback((destination: number) => {
    const unchanged = !textDirty;
    if (unchanged) {
      go(destination);
      return;
    }
    const decision = textDirty ? (editorText.trim() ? "keep" : "drop")
      : activeDecision === "unreviewed" ? "unsure" : activeDecision;
    void saveDocument(decision, destination);
  }, [activeDecision, editorText, go, saveDocument, textDirty]);

  const commitPageInput = useCallback(() => {
    if (busy || llmInFlight.current) return;
    const page = Number(pageInput);
    if (!Number.isInteger(page) || page < 1 || page > totalItems) {
      setPageInput(String(ordinal + 1));
      setStatus(`请输入 1 到 ${totalItems.toLocaleString()} 之间的条目序号`);
      return;
    }
    const destination = page - 1;
    if (destination === ordinal) return;
    saveAndGo(destination);
  }, [busy, ordinal, pageInput, saveAndGo, totalItems]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (showSetup || showSettings) return;
      // Save the current document on plain Up, even when a control has focus.
      if (event.key === "ArrowUp"
        && !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey
        && !event.isComposing) {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (!event.repeat && !busy && !llmInFlight.current) {
          void saveDocument("keep", ordinal);
        }
        return;
      }
      // Reserve plain left/right for document navigation before focused controls handle them.
      if ((event.key === "ArrowLeft" || event.key === "ArrowRight")
        && !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey
        && !event.isComposing) {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (!busy && !llmInFlight.current) {
          saveAndGo(ordinal + (event.key === "ArrowRight" ? 1 : -1));
        }
        return;
      }
      if (busy || llmInFlight.current) return;
      const target = event.target as HTMLElement;
      // Form fields keep their native undo and other shortcuts; the block editor uses our draft history.
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      const typing = target.isContentEditable;
      if (event.ctrlKey && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) redoDraft();
        else undoDraft();
        return;
      }
      if (event.ctrlKey && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redoDraft();
        return;
      }
      if (event.ctrlKey && event.key.toLowerCase() === "d") {
        event.preventDefault();
        void saveDocument("drop");
        return;
      }
      if (typing) return;
      if (event.key.toLowerCase() === "x" && activeBlockId) {
        updateDraftBlocks(draftBlocks.map((block) =>
          block.id === activeBlockId ? { ...block, deleted: true } : block
        ));
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [
    activeBlockId,
    busy,
    draftBlocks,
    ordinal,
    redoDraft,
    saveAndGo,
    saveDocument,
    showSetup,
    showSettings,
    undoDraft,
    updateDraftBlocks,
  ]);

  const changeProject = useCallback((nextProjectId: string) => {
    if (busy || llmInFlight.current) return;
    setProjectId(nextProjectId);
    setChosenQueueId("");
    setOrdinal(0);
  }, [busy]);

  return {
    projects,
    projectId,
    ordinal,
    pageInput,
    document,
    draftBlocks,
    editorText,
    textDirty,
    busy,
    llmCleaning,
    llmResult,
    status,
    activeBlockId,
    showSetup,
    selectedQueue,
    totalItems,
    cleanProgress,
    activeDecision,
    activeDecisionOrigin,
    setPageInput,
    setActiveBlockId,
    setShowSetup,
    finishSetup,
    updateDraftBlocks,
    cleanCurrentDocument,
    undoDraft,
    saveAndGo,
    commitPageInput,
    changeProject,
  };
}
