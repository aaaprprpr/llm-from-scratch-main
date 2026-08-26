import type { Project, Queue } from "../types";

type Props = {
  showSetup: boolean;
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
  ordinal: number;
  pageInput: string;
  onShowSetupChange: (showSetup: boolean) => void;
  onSimplify: () => void;
  onProjectChange: (projectId: string) => void;
  onPageInputChange: (value: string) => void;
  onCommitPage: () => void;
  onNavigate: (destination: number) => void;
};

export default function ReviewSidebar({
  showSetup,
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
  ordinal,
  pageInput,
  onShowSetupChange,
  onSimplify,
  onProjectChange,
  onPageInputChange,
  onCommitPage,
  onNavigate,
}: Props) {
  const completed = selectedQueue?.state_counts.done ?? 0;

  return (
    <aside className="sidebar panel">
      <nav className="sidebar-nav">
        <button className={!showSetup ? "active" : ""} onClick={() => onShowSetupChange(false)}>清洗</button>
        <button className={showSetup ? "active" : ""} onClick={() => onShowSetupChange(true)} disabled={textDirty}>导入数据</button>
      </nav>
      <div className={`status sidebar-status ${status.includes("失败") ? "error" : ""}`}>
        <span className="status-dot" />
        {status}
      </div>

      {!showSetup && <div className="review-sidebar-content">
        {hasDocument && <p className={`current-decision ${activeDecision}`}>
          当前状态：{decisionLabel}{textDirty ? " · 有未保存修改" : ""}
        </p>}
        {hasDocument && <button
          type="button"
          className="simplify-button"
          onClick={onSimplify}
          disabled={busy}
        >繁体转简体</button>}
        <label>
          项目
          <select value={projectId} onChange={(event) => onProjectChange(event.target.value)} disabled={textDirty}>
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
          </>
        )}

        <div className="pager">
          <button onClick={() => onNavigate(ordinal - 1)} disabled={ordinal <= 0 || busy}>←</button>
          <input
            type="number"
            min={1}
            max={Math.max(1, totalItems)}
            value={pageInput}
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
          ↑ 完成并保留 · → 跳过 · ← 返回<br />
          Ctrl D 丢弃整条<br />
          Ctrl Z 撤销 · Ctrl Y 重做<br />
          编辑正文时按 Esc 退出编辑
        </p>
      </div>}
    </aside>
  );
}
