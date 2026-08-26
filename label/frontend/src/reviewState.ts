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
  if (
    value.document_review?.edited_text !== null
    && value.document_review?.edited_text !== undefined
  ) {
    return editableBlocksFromText(value.materialized_text, value.document.doc_id);
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
