import { useEffect, useState } from "react";
import { requestJson } from "../api";

type Status = "uncleaned" | "manual" | "single_llm" | "batch" | "risk" | "rejected" | "failed" | "review";
type Filter = "all" | Status;
type Item = { ordinal: number; status: Status; decision: "keep" | "drop" | null };
type Page = { items: Item[]; next_start: number | null; total: number };

type Props = {
  queueId: string;
  currentOrdinal: number;
  busy: boolean;
  refreshToken: string;
  onNavigate: (ordinal: number) => void;
};

const labels: Record<Status, string> = {
  uncleaned: "未清理",
  manual: "人工清理",
  single_llm: "单条 AI",
  batch: "批量清理",
  risk: "敏感待人工",
  rejected: "接口拒绝待人工",
  failed: "清洗失败",
  review: "待复核",
};

export default function QueueStatusList({ queueId, currentOrdinal, busy, refreshToken, onNavigate }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const [start, setStart] = useState(() => Math.max(0, currentOrdinal - 10));
  const [history, setHistory] = useState<number[]>([]);
  const [page, setPage] = useState<Page | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setFilter("all");
    setStart(Math.max(0, currentOrdinal - 10));
    setHistory([]);
    setPage(null);
  }, [queueId]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    requestJson<Page>(`/api/queues/${queueId}/statuses?start=${start}&limit=60&status=${filter}`)
      .then((value) => { if (active) { setPage(value); setError(""); } })
      .catch((cause) => { if (active) setError(String(cause)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [queueId, start, filter, refreshToken]);

  useEffect(() => {
    if (filter !== "all" || !page || page.items.length === 0) return;
    const first = page.items[0].ordinal;
    const last = page.items[page.items.length - 1].ordinal;
    if (currentOrdinal < first || currentOrdinal > last) {
      setStart(Math.max(0, currentOrdinal - 10));
      setHistory([]);
    }
  }, [currentOrdinal, filter, page]);

  const changeFilter = (value: Filter) => {
    setFilter(value);
    setStart(value === "all" ? Math.max(0, currentOrdinal - 10) : 0);
    setHistory([]);
    setPage(null);
  };

  const previous = () => {
    const last = history[history.length - 1];
    if (last === undefined) return;
    setHistory(history.slice(0, -1));
    setStart(last);
  };

  return <section className="queue-status-section" aria-label="条目清洗状态">
    <div className="queue-status-heading">
      <strong>条目列表</strong>
      <button onClick={() => { setStart(Math.max(0, currentOrdinal - 10)); setHistory([]); }}
        disabled={busy} title="定位到当前条目">当前</button>
    </div>
    <select aria-label="筛选清洗状态" value={filter}
      onChange={(event) => changeFilter(event.target.value as Filter)}>
      <option value="all">全部状态</option>
      {Object.entries(labels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
    </select>
    <div className="queue-status-scroll">
      {error && <p className="queue-status-error">读取状态失败：{error}</p>}
      {loading && !page && <p className="queue-status-empty">正在读取…</p>}
      {page && page.items.length === 0 && <p className="queue-status-empty">没有符合条件的条目</p>}
      {page?.items.map((item) => <button key={item.ordinal}
        className={`queue-status-row ${item.status} ${item.ordinal === currentOrdinal ? "active" : ""}`}
        aria-current={item.ordinal === currentOrdinal ? "true" : undefined}
        disabled={busy} onClick={() => onNavigate(item.ordinal)}>
        <span>第 {item.ordinal + 1} 条</span>
        <span>{labels[item.status]}{item.decision === "drop" ? " · 丢弃" : item.decision === "keep" ? " · 保留" : ""}</span>
      </button>)}
    </div>
    <div className="queue-status-pager">
      <button onClick={previous} disabled={busy || loading || history.length === 0}>上一页</button>
      <span>{page?.items.length ? `${page.items[0].ordinal + 1} 起` : "—"}</span>
      <button onClick={() => {
        if (page?.next_start == null) return;
        setHistory([...history, start]);
        setStart(page.next_start);
      }} disabled={busy || loading || page?.next_start == null}>下一页</button>
    </div>
  </section>;
}
