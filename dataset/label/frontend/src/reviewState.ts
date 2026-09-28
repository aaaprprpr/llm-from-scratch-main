import {
  editableBlocksFromText,
  type EditableBlock,
} from "./BlockEditor";
import type { Queue, QueueDocument } from "./types";

export const decisionLabels: Record<string, string> = {
  unreviewed: "未处理",
  keep: "保留",
  drop: "丢弃",
  unsure: "待定",
};

export function queueSize(queue: Queue | null): number {
  if (!queue) return 0;
  return Object.values(queue.state_counts).reduce((sum, value) => sum + value, 0);
}

export function makeEditableBlocks(value: QueueDocument): EditableBlock[] {
  if (value.document_review?.decision === "drop") {
    const prior = value.document_review.edited_text;
    const blocks = prior != null
      ? editableBlocksFromText(prior, value.document.doc_id)
      : value.blocks.map((block) => ({
          id: block.block_id,
          sourceOrdinal: block.ordinal,
          text: block.text,
          separatorAfter: block.separator_after,
          deleted: false,
        }));
    return blocks.map((block) => ({ ...block, deleted: true }));
  }
  if (value.effective_source === "manual" && value.document_review?.edited_text != null) {
    return editableBlocksFromText(value.materialized_text, value.document.doc_id);
  }
  if (value.effective_source === "batch") {
    if (value.batch_clean?.status === "keep") {
      return editableBlocksFromText(value.materialized_text, value.document.doc_id);
    }
    if (value.batch_clean?.status === "drop") {
      return value.blocks.map((block) => ({
        id: block.block_id,
        sourceOrdinal: block.ordinal,
        text: block.text,
        separatorAfter: block.separator_after,
        deleted: true,
      }));
    }
  }
  return value.blocks.map((block) => ({
    id: block.block_id,
    sourceOrdinal: block.ordinal,
    text: block.text,
    separatorAfter: block.separator_after,
    deleted: block.review?.decision === "drop",
  }));
}

export function cloneBlocks(blocks: EditableBlock[]): EditableBlock[] {
  return blocks.map((block) => ({ ...block }));
}
