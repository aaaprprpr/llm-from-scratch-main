import BlockEditor, { type EditAction, type EditableBlock } from "../BlockEditor";
import SetupWizard from "../SetupWizard";
import type { Project, QueueDocument } from "../types";

type Props = {
  showSetup: boolean;
  projects: Project[];
  projectId: string;
  document: QueueDocument | null;
  draftBlocks: EditableBlock[];
  editorText: string;
  textDirty: boolean;
  busy: boolean;
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
            <strong className={textDirty ? "dirty" : ""}>{textDirty ? "尚未保存" : "已保存"}</strong>
          </div>
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
