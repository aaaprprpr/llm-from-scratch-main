import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { requestJson } from "../api";
import {
  renderEditableBlocks,
  type EditAction,
  type EditableBlock,
} from "../BlockEditor";
import {
  cloneBlocks,
  decisionLabels,
  makeEditableBlocks,
  queueSize,
} from "../reviewState";
import type { Project, ProjectDetail, QueueDocument } from "../types";

export function useReviewWorkspace() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState("");
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [ordinal, setOrdinal] = useState(0);
  const [pageInput, setPageInput] = useState("1");
  const [document, setDocument] = useState<QueueDocument | null>(null);
  const [draftBlocks, setDraftBlocks] = useState<EditableBlock[]>([]);
  const [editorText, setEditorText] = useState("");
  const [textDirty, setTextDirty] = useState(false);
  const [quality, setQuality] = useState<number | null>(null);
  const [category, setCategory] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("正在连接后端…");
  const [activeBlockId, setActiveBlockId] = useState<string | null>(null);
  const undoStack = useRef<EditableBlock[][]>([]);
  const redoStack = useRef<EditableBlock[][]>([]);
  const typingGroup = useRef<{ blockId: string; at: number } | null>(null);
  const [showSetup, setShowSetup] = useState(false);
  const [projectRefresh, setProjectRefresh] = useState(0);

  const selectedQueue = useMemo(() => {
    const fullQueues = project?.queues.filter(
      (queue) => queue.sampling_policy.type === "full_dataset",
    ) ?? [];
    return fullQueues[fullQueues.length - 1] ?? null;
  }, [project]);
  const queueId = selectedQueue?.queue_id ?? "";
  const totalItems = queueSize(selectedQueue);
  const activeDecision = document?.document_review?.decision ?? "unreviewed";

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
      .then((value) => setProject(value))
      .catch((error) => setStatus(`读取项目失败：${String(error)}`));
  }, [projectId, projectRefresh]);

  const finishSetup = useCallback(async (nextProjectId: string, _nextQueueId: string) => {
    await loadProjects();
    setProjectId(nextProjectId);
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
      const review = value.document_review;
      setQuality(review?.quality ?? null);
      setCategory(review?.primary_category ?? "");
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

  useEffect(() => {
    setPageInput(String(ordinal + 1));
  }, [ordinal]);

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
    if (!document) return;
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
          quality,
          primary_category: category || null,
          flags: document.document_review?.flags ?? [],
          notes,
          edited_text: editorText,
        }),
      });
      setTextDirty(false);
      setStatus(`已保存：${decisionLabels[decision]}`);
      if (destination >= 0 && destination < totalItems && destination !== ordinal) {
        go(destination);
      } else {
        await loadDocument();
      }
    } catch (error) {
      setStatus(`保存失败：${String(error)}`);
      await loadDocument();
    } finally {
      setBusy(false);
    }
  }, [
    category,
    document,
    editorText,
    go,
    loadDocument,
    notes,
    ordinal,
    quality,
    queueId,
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

  const simplifyCurrentDocument = useCallback(async () => {
    const keptBlocks = draftBlocks.filter((block) => !block.deleted);
    if (!keptBlocks.length) {
      setStatus("当前清洗结果为空，没有可转换的正文");
      return;
    }
    setBusy(true);
    setStatus("正在将当前正文中的繁体字转换为简体字…");
    try {
      const result = await requestJson<{ texts: string[]; changed_characters: number }>(
        "/api/text/simplify",
        {
          method: "POST",
          body: JSON.stringify({ texts: keptBlocks.map((block) => block.text) }),
        },
      );
      let convertedIndex = 0;
      const nextBlocks = draftBlocks.map((block) => block.deleted ? block : {
        ...block,
        text: result.texts[convertedIndex++],
      });
      if (result.changed_characters > 0) {
        updateDraftBlocks(nextBlocks, { kind: "command" });
        setStatus(`已转换 ${result.changed_characters.toLocaleString()} 个字符，尚未保存`);
      } else {
        setStatus("当前正文没有需要转换的繁体字");
      }
    } catch (error) {
      setStatus(`繁体转简体失败：${String(error)}`);
    } finally {
      setBusy(false);
    }
  }, [draftBlocks, updateDraftBlocks]);

  const undoDraft = useCallback(() => {
    const previous = undoStack.current.pop();
    if (!previous) return;
    redoStack.current.push(cloneBlocks(draftBlocks));
    typingGroup.current = null;
    applyDraftBlocks(previous);
    setStatus("已撤销本次文档修改");
  }, [applyDraftBlocks, draftBlocks]);

  const redoDraft = useCallback(() => {
    const next = redoStack.current.pop();
    if (!next) return;
    undoStack.current.push(cloneBlocks(draftBlocks));
    typingGroup.current = null;
    applyDraftBlocks(next);
    setStatus("已重做本次文档修改");
  }, [applyDraftBlocks, draftBlocks]);

  const saveAndGo = useCallback((destination: number) => {
    const decision = activeDecision === "unreviewed" ? "unsure" : activeDecision;
    void saveDocument(decision, destination);
  }, [activeDecision, saveDocument]);

  const commitPageInput = useCallback(() => {
    const page = Number(pageInput);
    if (!Number.isInteger(page) || page < 1 || page > totalItems) {
      setPageInput(String(ordinal + 1));
      setStatus(`请输入 1 到 ${totalItems.toLocaleString()} 之间的条目序号`);
      return;
    }
    const destination = page - 1;
    if (destination === ordinal) return;
    saveAndGo(destination);
  }, [ordinal, pageInput, saveAndGo, totalItems]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      const typing = target.isContentEditable
        || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
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
      if (typing || busy) return;
      if (/^[0-3]$/.test(event.key)) setQuality(Number(event.key));
      if (event.key.toLowerCase() === "x" && activeBlockId) {
        updateDraftBlocks(draftBlocks.map((block) =>
          block.id === activeBlockId ? { ...block, deleted: true } : block
        ));
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        void saveDocument("keep");
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        saveAndGo(ordinal + 1);
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        saveAndGo(ordinal - 1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    activeBlockId,
    busy,
    draftBlocks,
    ordinal,
    redoDraft,
    saveAndGo,
    saveDocument,
    undoDraft,
    updateDraftBlocks,
  ]);

  const changeProject = useCallback((nextProjectId: string) => {
    setProjectId(nextProjectId);
    setOrdinal(0);
  }, []);

  return {
    projects,
    projectId,
    ordinal,
    pageInput,
    document,
    draftBlocks,
    editorText,
    textDirty,
    quality,
    category,
    notes,
    busy,
    status,
    activeBlockId,
    showSetup,
    selectedQueue,
    totalItems,
    activeDecision,
    setPageInput,
    setQuality,
    setCategory,
    setNotes,
    setActiveBlockId,
    setShowSetup,
    finishSetup,
    updateDraftBlocks,
    simplifyCurrentDocument,
    saveAndGo,
    commitPageInput,
    changeProject,
  };
}
