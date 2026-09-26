import { useEffect, useRef, useState } from "react";
import type { TextDiff } from "../textDiff";

type Props = { original: string; draft: string };

export default function DraftDiff({ original, draft }: Props) {
  const [mode, setMode] = useState<"diff" | "original" | "draft">("diff");
  const [diff, setDiff] = useState<{ original: string; draft: string; result: TextDiff } | null>(null);
  const [failed, setFailed] = useState(false);
  const worker = useRef<Worker | null>(null);
  const latest = useRef({ id: 0, original, draft });

  useEffect(() => {
    const instance = new Worker(new URL("../workers/textDiff.worker.ts", import.meta.url), { type: "module" });
    worker.current = instance;
    instance.onmessage = (event: MessageEvent<{ id: number; result: TextDiff }>) => {
      if (event.data.id !== latest.current.id) return;
      setDiff({ original: latest.current.original, draft: latest.current.draft, result: event.data.result });
    };
    instance.onerror = () => setFailed(true);
    return () => { worker.current = null; instance.terminate(); };
  }, []);

  useEffect(() => {
    const request = { id: latest.current.id + 1, original, draft };
    latest.current = request;
    const timer = window.setTimeout(() => worker.current?.postMessage(request), 100);
    return () => window.clearTimeout(timer);
  }, [original, draft]);

  const result = diff?.original === original && diff?.draft === draft ? diff.result : null;
  const same = original === draft;
  return <section className="raw-preview draft-diff" aria-label="简体原文与当前草稿对照">
    <h3>简体原文与当前草稿</h3>
    <div className="diff-tabs" role="group" aria-label="预览方式">
      <button className={mode === "diff" ? "active" : ""} onClick={() => setMode("diff")}>删改对照</button>
      <button className={mode === "original" ? "active" : ""} onClick={() => setMode("original")}>简体原文</button>
      <button className={mode === "draft" ? "active" : ""} onClick={() => setMode("draft")}>拼接结果</button>
    </div>
    {mode === "diff" && <>
      <p className="diff-legend"><span className="diff-removed">删除 / 移出</span><span className="diff-added">插入 / 移入</span></p>
      <p className="diff-status" aria-live="polite">{same ? "与简体原文一致" : result ? `−${result.removed} 字符 / +${result.added} 字符${result.coarse ? " · 大范围改动按行展示" : ""}` : failed ? "差异计算失败，可切换原文与拼接结果查看" : "正在更新删改对照…"}</p>
    </>}
    <pre className="diff-text" aria-label={mode === "original" ? "简体原文" : mode === "draft" ? "当前拼接结果" : "删改差异"}>
      {mode === "original" ? original : mode === "draft" ? draft : same ? original : result ? result.parts.map((part, index) => {
        const label = part.kind === "removed" ? "删除 / 移出" : "插入 / 移入";
        const content = part.kind !== "equal" && !part.value.trim()
          ? part.value.replace(/ /g, "·").replace(/\t/g, "⇥").replace(/\r/g, "␍").replace(/\n/g, "↵\n")
          : part.value;
        return part.kind === "equal" ? <span key={index}>{content}</span>
          : part.kind === "removed" ? <del className="diff-removed" key={index} title={part.move ? `移出片段 ${part.move}` : label}>{content}</del>
          : <ins className="diff-added" key={index} title={part.move ? `移入片段 ${part.move}` : label}>{content}</ins>;
      }) : original}
    </pre>
  </section>;
}
