import BlockEditor, { type EditAction, type EditableBlock } from "../BlockEditor";
import SetupWizard from "../SetupWizard";
import type { LlmCleaningResult, Project, QueueDocument } from "../types";

type Props = {
  showSetup: boolean;
  projects: Project[];
  document: QueueDocument | null;
  draftBlocks: EditableBlock[];
  editorText: string;
  textDirty: boolean;
  busy: boolean;
  llmCleaning: boolean;
  llmResult: LlmCleaningResult | null;
  onLlmClean: () => void;
  activeBlockId: string | null;
  onFinishSetup: (projectId: string, queueId: string) => Promise<void>;
  onShowSetup: () => void;
  onActiveBlockChange: (blockId: string) => void;
  onBlocksChange: (blocks: EditableBlock[], action: EditAction) => void;
};

export default function DocumentWorkspace({
  showSetup,
  projects,
  document,
  draftBlocks,
  editorText,
  textDirty,
  busy,
  llmCleaning,
  llmResult,
  onLlmClean,
  activeBlockId,
  onFinishSetup,
  onShowSetup,
  onActiveBlockChange,
  onBlocksChange,
}: Props) {
  return (
    <main className="document-panel panel">
      {showSetup ? (
        <SetupWizard
          projects={projects}
          onComplete={onFinishSetup}
        />
      ) : !document ? (
        <div className="empty-state">
          <span>⌁</span>
          <h2>当前项目还没有全量清洗数据</h2>
          <p>可以先导入本地数据集目录、逐行 JSON 文件或纯文本文件。</p>
          <button className="primary" onClick={onShowSetup}>导入数据集</button>
        </div>
      ) : (
        <>
          <div className="editor-summary">
            <span>
              {draftBlocks.filter((block) => !block.deleted).length.toLocaleString()} 段
              {` · ${Array.from(editorText).length.toLocaleString()} 字符`}
            </span>
            <strong className={textDirty ? "dirty" : ""}>{textDirty ? "尚未保存" : "已保存"}</strong>
            <button className="primary llm-clean-button" onClick={onLlmClean} disabled={busy}>
              {llmCleaning ? "自动清洗中…" : "自动清洗并保存本条"}
            </button>
          </div>
          {llmResult && <section className="llm-result" aria-label="LLM 清洗结果">
            <strong>{llmResult.saved ? "清洗结果已直接保存" : "存在未完成分块，未写入清洗结果"}</strong>
            {llmResult.requests !== undefined && (
              <p>本次模型请求 {llmResult.requests} 次 · 条内分块 {llmResult.chunks} 块</p>
            )}
            {llmResult.risk_fallback_chunks?.length ? (
              <p>DeepSeek 敏感拒绝后，{llmResult.risk_fallback_chunks.length} 块改由 {llmResult.risk_fallback_chunks[0].model} 清洗</p>
            ) : null}
            {llmResult.saved && <p>删除 {llmResult.removals.length} 处 · 修订 {llmResult.edits.length} 处</p>}
            {!llmResult.saved && llmResult.warnings.slice(0, 3).map((warning, index) => (
              <p className="llm-warning" key={index}>{warning}</p>
            ))}
            {!llmResult.saved && llmResult.warnings.length > 3 && (
              <p>另有 {llmResult.warnings.length - 3} 块未完成</p>
            )}
          </section>}
          <BlockEditor
            blocks={draftBlocks}
            busy={busy}
            activeBlockId={activeBlockId}
            onActiveBlockChange={onActiveBlockChange}
            onChange={onBlocksChange}
          />
        </>
      )}
    </main>
  );
}
