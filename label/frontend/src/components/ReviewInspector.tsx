import type { QueueDocument } from "../types";
import { categories } from "../reviewState";
import DraftDiff from "./DraftDiff";

type Props = {
  document: QueueDocument | null;
  editorText: string;
  quality: number | null;
  category: string;
  notes: string;
  onQualityChange: (quality: number | null) => void;
  onCategoryChange: (category: string) => void;
  onNotesChange: (notes: string) => void;
};

export default function ReviewInspector({
  document,
  editorText,
  quality,
  category,
  notes,
  onQualityChange,
  onCategoryChange,
  onNotesChange,
}: Props) {
  return (
    <aside className="provenance panel">
      {document ? (
        <>
          <section className="optional-labels">
            <div className="quality-sidebar">
              <span>质量评分（可选）</span>
              <div>
                <button className={quality === null ? "active" : ""} onClick={() => onQualityChange(null)}>不填</button>
                {[0, 1, 2, 3].map((value) => (
                  <button key={value} className={quality === value ? "active" : ""} onClick={() => onQualityChange(value)}>{value}</button>
                ))}
              </div>
            </div>
            <label>内容类型（可选）
              <select value={category} onChange={(event) => onCategoryChange(event.target.value)}>
                <option value="">不填写</option>
                {categories.map((value) => <option key={value.value} value={value.value}>{value.label}</option>)}
              </select>
            </label>
            <label>备注（可选）
              <textarea value={notes} onChange={(event) => onNotesChange(event.target.value)} rows={4} />
            </label>
          </section>

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

          <DraftDiff key={document.document.doc_id} original={document.simplified.raw_text} draft={editorText} />
        </>
      ) : <p className="muted">选择文档后显示清洗标签和来源信息。</p>}
    </aside>
  );
}
