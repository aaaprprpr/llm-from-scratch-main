import BlockEditor, { type EditAction, type EditableBlock } from "../BlockEditor";
import SetupWizard from "../SetupWizard";
import { categories, decisionLabels } from "../reviewState";
import type { LlmCleaningResult, Project, QueueDocument } from "../types";

const removalReasons: Record<string, string> = {
  advertisement: "广告推广", navigation: "导航", empty_section: "空栏目",
  repetition: "重复模板", garbled: "乱码", unrelated: "网页残留",
};

type Props = {
  showSetup: boolean;
  projects: Project[];
  projectId: string;
  document: QueueDocument | null;
  draftBlocks: EditableBlock[];
  editorText: string;
  textDirty: boolean;
  busy: boolean;
  llmCleaning: boolean;
  llmResult: LlmCleaningResult | null;
  onLlmClean: () => void;
  onUndoLlmClean: () => void;
  activeBlockId: string | null;
  onFinishSetup: (projectId: string, queueId: string) => Promise<void>;
  onShowSetup: () => void;
  onActiveBlockChange: (blockId: string) => void;
  onBlocksChange: (blocks: EditableBlock[], action: EditAction) => void;
};

export default function DocumentWorkspace({
  showSetup,
  projects,
  projectId,
  document,
  draftBlocks,
  editorText,
  textDirty,
  busy,
  llmCleaning,
  llmResult,
  onLlmClean,
  onUndoLlmClean,
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
          initialProjectId={projectId}
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
            <strong className={textDirty ? "dirty" : ""}>{textDirty ? "尚未保存" : document.materialized_text !== document.simplified.materialized_text ? "已自动转简体" : "已保存"}</strong>
            <button className="primary llm-clean-button" onClick={onLlmClean} disabled={busy}>
              {llmCleaning ? "本地模型清洗中…" : "LLM 清洗本条"}
            </button>
          </div>
          {llmResult && <section className="llm-result" aria-label="本地模型清洗建议">
            <div className="llm-result-heading">
              <strong>清洗草稿已返回 · {decisionLabels[llmResult.decision]}</strong>
              {llmResult.text_changed && <button className="secondary" disabled={busy} onClick={onUndoLlmClean}>撤销本次清洗</button>}
            </div>
            <p>删除 {llmResult.removals.length} 处 · 修订 {llmResult.edits.length} 处 · 重组 {llmResult.reordered_chunks} 块 · {llmResult.elapsed_seconds} 秒</p>
            <p>可直接改字、选区删除、拖动片段拼接和重排段落。右侧实时对照原文；↑ 保存完成的草稿，Ctrl D 丢弃整条。</p>
            <p>参考评分 {llmResult.quality}/3 · {categories.find((item) => item.value === llmResult.category)?.label ?? llmResult.category}</p>
            {llmResult.warnings.map((warning) => <p className="llm-warning" key={warning}>{warning}</p>)}
            <details open={llmResult.assessments.length === 1}>
              <summary>查看判断理由</summary>
              {llmResult.assessments.map((item) => <p key={item.chunk}>{llmResult.assessments.length > 1 ? `第 ${item.chunk} 块：` : ""}{item.summary}</p>)}
            </details>
            {llmResult.removals.length > 0 && <details>
              <summary>查看建议删除的原文（{llmResult.removals.length} 处）</summary>
              {llmResult.removals.map((item, index) => <div className="llm-removal" key={index}>
                <strong>{removalReasons[item.reason] ?? item.reason}</strong>
                <pre>{item.text}</pre>
              </div>)}
            </details>}
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
