import type { Project, Queue } from "../types";
import QueueStatusList from "./QueueStatusList";

type Props = {
  showSetup: boolean;
  showSettings: boolean;
  textDirty: boolean;
  status: string;
  hasDocument: boolean;
  activeDecision: string;
  decisionLabel: string;
  busy: boolean;
  projects: Project[];
  projectId: string;
  selectedQueue: Queue | null;
  totalItems: number;
  cleanProgress: { completed: number; manual_completed: number; manual_llm_saved: number;
    batch_counts: { keep?: number; drop?: number; incomplete?: number } } | null;
  ordinal: number;
  pageInput: string;
  statusRefresh: string;
  onShowSetupChange: (showSetup: boolean) => void;
  onShowSettingsChange: (showSettings: boolean) => void;
  onProjectChange: (projectId: string) => void;
  onPageInputChange: (value: string) => void;
  onCommitPage: () => void;
  onNavigate: (destination: number) => void;
};

export default function ReviewSidebar({
  showSetup,
  showSettings,
  textDirty,
  status,
  hasDocument,
  activeDecision,
  decisionLabel,
  busy,
  projects,
  projectId,
  selectedQueue,
  totalItems,
  cleanProgress,
  ordinal,
  pageInput,
  statusRefresh,
  onShowSetupChange,
  onShowSettingsChange,
  onProjectChange,
  onPageInputChange,
  onCommitPage,
  onNavigate,
}: Props) {
  const completed = cleanProgress?.completed ?? 0;
  const batchCount = (cleanProgress?.batch_counts.keep ?? 0)
    + (cleanProgress?.batch_counts.drop ?? 0)
    + (cleanProgress?.batch_counts.incomplete ?? 0);

  return (
    <aside className="sidebar panel">
      <nav className="sidebar-nav">
        <button className={!showSetup && !showSettings ? "active" : ""} onClick={() => { onShowSettingsChange(false); onShowSetupChange(false); }} disabled={busy}>清洗</button>
        <button className={showSetup ? "active" : ""} onClick={() => onShowSetupChange(true)} disabled={textDirty || busy} title={textDirty ? "请先保存或撤销当前正文修改，再进入导入页" : undefined}>导入数据</button>
        <button className={showSettings ? "active" : ""} onClick={() => onShowSettingsChange(true)} disabled={busy}>设置</button>
      </nav>
      {(status.includes("失败") || status.startsWith("无法")) && <p className="sidebar-error">{status}</p>}

      {!showSetup && !showSettings && <div className="review-sidebar-content">
        {hasDocument && <p className={`current-decision ${activeDecision}`}>
          当前状态：{decisionLabel}{textDirty ? " · 有未保存修改" : ""}
        </p>}
        <label>
          项目
          <select value={projectId} onChange={(event) => onProjectChange(event.target.value)} disabled={textDirty || busy}>
            <option value="">选择项目</option>
            {projects.map((value) => (
              <option key={value.project_id} value={value.project_id}>
                {value.name}
              </option>
            ))}
          </select>
        </label>
        {selectedQueue && (
          <>
            <div className="progress-copy">
              <span>清洗进度</span>
              <strong>{completed.toLocaleString()} / {totalItems.toLocaleString()}</strong>
            </div>
            <div className="progress-track">
              <span style={{ width: `${totalItems ? (completed / totalItems) * 100 : 0}%` }} />
            </div>
            <p className="progress-breakdown">人工确认 {cleanProgress?.manual_completed ?? 0} · 单条 LLM {cleanProgress?.manual_llm_saved ?? 0} · 批量处理 {batchCount}</p>
          </>
        )}

        <div className="pager">
          <button onClick={() => onNavigate(ordinal - 1)} disabled={ordinal <= 0 || busy}>←</button>
          <input
            type="number"
            min={1}
            max={Math.max(1, totalItems)}
            value={pageInput}
            disabled={busy}
            onChange={(event) => onPageInputChange(event.target.value)}
            onBlur={onCommitPage}
            onKeyDown={(event) => {
              if (event.key === "Enter") event.currentTarget.blur();
              if (event.key === "Escape") {
                onPageInputChange(String(ordinal + 1));
                window.requestAnimationFrame(() => event.currentTarget.blur());
              }
            }}
            aria-label="跳转到条目"
          />
          <button onClick={() => onNavigate(ordinal + 1)} disabled={ordinal + 1 >= totalItems || busy}>→</button>
        </div>
        <p className="shortcut-help">
          ↑ 完成并保留 · ← 上一条 · → 下一条<br />
          左右键在输入框和编辑区也翻页<br />
          Ctrl D 丢弃整条<br />
          Ctrl Z 撤销 · Ctrl Y 重做<br />
          编辑正文时按 Esc 退出编辑
        </p>
        {selectedQueue && <QueueStatusList
          queueId={selectedQueue.queue_id}
          currentOrdinal={ordinal}
          busy={busy}
          refreshToken={statusRefresh}
          onNavigate={onNavigate}
        />}
      </div>}
    </aside>
  );
}
