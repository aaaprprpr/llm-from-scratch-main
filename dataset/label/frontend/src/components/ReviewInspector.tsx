import type { QueueDocument } from "../types";
import DraftDiff, { type DiffView } from "./DraftDiff";

type Props = {
  document: QueueDocument | null;
  editorText: string;
  diffView: DiffView;
  onDiffViewChange: (mode: DiffView) => void;
};

export default function ReviewInspector({
  document,
  editorText,
  diffView,
  onDiffViewChange,
}: Props) {
  return (
    <aside className="provenance panel">
      {document ? (
        <>
          <details className="source-details">
            <summary>来源信息</summary>
            <dl>
              <div><dt>来源</dt><dd>{document.provenance.source_id}</dd></div>
              <div><dt>来源行号</dt><dd>{document.provenance.source_row.toLocaleString()}</dd></div>
              <div><dt>原始标识</dt><dd>{document.provenance.source_local_id || "—"}</dd></div>
              <div><dt>数据版本</dt><dd className="mono wrap">{document.provenance.source_revision}</dd></div>
              <div><dt>原始路径</dt><dd className="wrap">{document.provenance.original_location}</dd></div>
              {document.provenance.url && (
                <div><dt>网页地址</dt><dd><a href={document.provenance.url} target="_blank" rel="noreferrer">打开来源 ↗</a></dd></div>
              )}
            </dl>
          </details>

          <DraftDiff original={document.review_text} draft={editorText} mode={diffView} onModeChange={onDiffViewChange} />
        </>
      ) : <p className="muted">选择文档后显示来源信息和修改对照。</p>}
    </aside>
  );
}
