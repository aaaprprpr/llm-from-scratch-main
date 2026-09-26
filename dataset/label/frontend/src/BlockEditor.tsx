import { useEffect, useLayoutEffect, useRef, useState } from "react";

export type EditableBlock = {
  id: string;
  sourceOrdinal: number | null;
  text: string;
  separatorAfter: string;
  deleted: boolean;
};

export type EditAction =
  | { kind: "typing"; blockId: string }
  | { kind: "command" };

type TextSelection = {
  blockId: string;
  start: number;
  end: number;
  text: string;
  left: number;
  top: number;
};

type SelectionDrag = Pick<TextSelection, "blockId" | "start" | "end" | "text">;

type Props = {
  blocks: EditableBlock[];
  busy: boolean;
  activeBlockId: string | null;
  onActiveBlockChange: (blockId: string) => void;
  onChange: (blocks: EditableBlock[], action: EditAction) => void;
};

const selectionHighlightName = "curation-text-selection";

function splitTextWithSeparators(text: string): Array<{ text: string; separatorAfter: string }> {
  const separator = /\n[ \t]*\n(?:[ \t]*\n)*/g;
  const parts: Array<{ text: string; separatorAfter: string }> = [];
  let cursor = 0;
  for (const match of text.matchAll(separator)) {
    const start = match.index ?? cursor;
    const value = text.slice(cursor, start);
    if (value) parts.push({ text: value, separatorAfter: match[0] });
    cursor = start + match[0].length;
  }
  const tail = text.slice(cursor);
  if (tail) parts.push({ text: tail, separatorAfter: "" });
  return parts;
}

export function editableBlocksFromText(text: string, docId: string): EditableBlock[] {
  return splitTextWithSeparators(text).map((part, index) => ({
    id: `${docId}:edited:${index}`,
    sourceOrdinal: index,
    text: part.text,
    separatorAfter: part.separatorAfter,
    deleted: false,
  }));
}

export function serializeEditableBlocks(blocks: EditableBlock[]) {
  const kept = blocks.filter((block) => !block.deleted);
  const untouchedOrder = blocks.every(
    (block, index) => !block.deleted && (block.sourceOrdinal === null || block.sourceOrdinal === index),
  );
  return kept.map((block, index) => ({ id: block.id, text: block.text,
    separator_after: untouchedOrder ? block.separatorAfter : index < kept.length - 1 ? "\n\n" : "" }));
}

export function renderEditableBlocks(blocks: EditableBlock[]): string {
  return serializeEditableBlocks(blocks).map((block) => block.text + block.separator_after).join("");
}

function textOffset(root: HTMLElement, node: Node, offset: number): number {
  const range = window.document.createRange();
  range.selectNodeContents(root);
  range.setEnd(node, offset);
  return range.toString().length;
}

function caretOffsetAtPoint(root: HTMLElement, x: number, y: number): number {
  const browserDocument = window.document as Document & {
    caretPositionFromPoint?: (x: number, y: number) => { offsetNode: Node; offset: number } | null;
    caretRangeFromPoint?: (x: number, y: number) => Range | null;
  };
  const position = browserDocument.caretPositionFromPoint?.(x, y);
  if (position && root.contains(position.offsetNode)) {
    return textOffset(root, position.offsetNode, position.offset);
  }
  const range = browserDocument.caretRangeFromPoint?.(x, y);
  if (range && root.contains(range.startContainer)) {
    return textOffset(root, range.startContainer, range.startOffset);
  }
  return (root.textContent ?? "").length;
}

function setCaret(root: HTMLElement, offset: number) {
  const walker = window.document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let remaining = Math.max(0, offset);
  let node = walker.nextNode();
  while (node) {
    const length = node.textContent?.length ?? 0;
    if (remaining <= length) {
      const range = window.document.createRange();
      range.setStart(node, remaining);
      range.collapse(true);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
      return;
    }
    remaining -= length;
    node = walker.nextNode();
  }
  root.focus();
}

function putPlainTextAtCaret(root: HTMLElement, text: string) {
  const selection = window.getSelection();
  if (!selection?.rangeCount) return;
  const range = selection.getRangeAt(0);
  if (!root.contains(range.commonAncestorContainer)) return;
  range.deleteContents();
  const node = window.document.createTextNode(text);
  range.insertNode(node);
  range.setStartAfter(node);
  range.collapse(true);
  selection.removeAllRanges();
  selection.addRange(range);
  root.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
}

function clearPersistentHighlight() {
  const registry = (CSS as unknown as {
    highlights?: { delete: (name: string) => void };
  }).highlights;
  registry?.delete(selectionHighlightName);
}

function persistHighlight(range: Range) {
  clearPersistentHighlight();
  const registry = (CSS as unknown as {
    highlights?: { set: (name: string, value: unknown) => void };
  }).highlights;
  const HighlightClass = (window as unknown as {
    Highlight?: new (...ranges: Range[]) => unknown;
  }).Highlight;
  if (registry && HighlightClass) {
    registry.set(selectionHighlightName, new HighlightClass(range.cloneRange()));
  }
}

function EditableText({
  block,
  disabled,
  selection,
  onActivate,
  onTextChange,
  onClearSelection,
  onSelectionConsumed,
  onSelection,
  onSelectionDragStart,
  onSelectionDrop,
}: {
  block: EditableBlock;
  disabled: boolean;
  selection: TextSelection | null;
  onActivate: () => void;
  onTextChange: (text: string, kind: "typing" | "command") => void;
  onClearSelection: () => void;
  onSelectionConsumed: () => void;
  onSelection: (root: HTMLElement) => void;
  onSelectionDragStart: (event: React.DragEvent<HTMLElement>) => void;
  onSelectionDrop: (event: React.DragEvent<HTMLElement>, targetOffset: number) => void;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    const root = rootRef.current;
    if (root && root.textContent !== block.text) root.textContent = block.text;
  }, [block.text]);

  return (
    <div
      ref={rootRef}
      className="editable-block-text"
      contentEditable={!disabled}
      suppressContentEditableWarning
      spellCheck={false}
      role="textbox"
      aria-multiline="true"
      aria-label="可编辑段落"
      onFocus={onActivate}
      onPointerDown={onActivate}
      onPointerUp={() => {
        window.requestAnimationFrame(() => {
          if (rootRef.current) onSelection(rootRef.current);
        });
      }}
      onInput={(event) => {
        if (selection) onSelectionConsumed();
        onTextChange(event.currentTarget.textContent ?? "", "typing");
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          onClearSelection();
          event.currentTarget.blur();
          return;
        }
        if (selection?.blockId === block.id && (event.key === "Backspace" || event.key === "Delete")) {
          event.preventDefault();
          const next = block.text.slice(0, selection.start) + block.text.slice(selection.end);
          onTextChange(next, "command");
          onSelectionConsumed();
          window.requestAnimationFrame(() => {
            if (rootRef.current) setCaret(rootRef.current, selection.start);
          });
          return;
        }
        if (event.key === "Enter") {
          event.preventDefault();
          putPlainTextAtCaret(event.currentTarget, "\n");
        }
      }}
      onPaste={(event) => {
        event.preventDefault();
        putPlainTextAtCaret(event.currentTarget, event.clipboardData.getData("text/plain"));
      }}
      onDragStart={onSelectionDragStart}
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes("application/x-curation-selection")) {
          event.preventDefault();
          event.dataTransfer.dropEffect = "move";
        }
      }}
      onDrop={(event) => {
        if (!event.dataTransfer.types.includes("application/x-curation-selection")) return;
        event.preventDefault();
        onSelectionDrop(
          event,
          caretOffsetAtPoint(event.currentTarget, event.clientX, event.clientY),
        );
      }}
    />
  );
}

export default function BlockEditor({
  blocks,
  busy,
  activeBlockId,
  onActiveBlockChange,
  onChange,
}: Props) {
  const [selection, setSelection] = useState<TextSelection | null>(null);
  const draggedSelection = useRef<SelectionDrag | null>(null);
  const draggedBlockId = useRef<string | null>(null);

  useEffect(() => () => clearPersistentHighlight(), []);
  useEffect(() => {
    if (!busy) return;
    clearPersistentHighlight();
    setSelection(null);
    draggedSelection.current = null;
    draggedBlockId.current = null;
  }, [busy]);

  const clearSelection = () => {
    clearPersistentHighlight();
    window.getSelection()?.removeAllRanges();
    setSelection(null);
  };

  const consumeSelection = () => {
    clearPersistentHighlight();
    setSelection(null);
  };

  const captureSelection = (root: HTMLElement) => {
    if (busy) return;
    const browserSelection = window.getSelection();
    if (!browserSelection?.rangeCount || browserSelection.isCollapsed) {
      consumeSelection();
      return;
    }
    const range = browserSelection.getRangeAt(0);
    if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) {
      clearSelection();
      return;
    }
    const start = textOffset(root, range.startContainer, range.startOffset);
    const end = textOffset(root, range.endContainer, range.endOffset);
    if (end <= start) {
      clearSelection();
      return;
    }
    const rect = range.getBoundingClientRect();
    const blockId = root.closest<HTMLElement>("[data-block-id]")?.dataset.blockId;
    if (!blockId) return;
    persistHighlight(range);
    setSelection({
      blockId,
      start,
      end,
      text: range.toString(),
      left: Math.min(window.innerWidth - 230, Math.max(12, rect.left + rect.width / 2 - 105)),
      top: Math.max(12, rect.top - 46),
    });
  };

  const replaceSelectedText = (replacement: string) => {
    if (!selection || busy) return;
    onChange(blocks.map((block) => block.id === selection.blockId ? {
      ...block,
      text: block.text.slice(0, selection.start) + replacement + block.text.slice(selection.end),
    } : block), { kind: "command" });
    clearSelection();
  };

  const startSelectionDrag = (event: React.DragEvent<HTMLElement>) => {
    if (!selection || busy) {
      event.preventDefault();
      return;
    }
    draggedSelection.current = selection;
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("application/x-curation-selection", "move");
    event.dataTransfer.setData("text/plain", selection.text);
  };

  const dropSelection = (targetBlockId: string, targetOffset: number) => {
    if (busy) return;
    const source = draggedSelection.current;
    if (!source) return;
    if (
      source.blockId === targetBlockId
      && targetOffset >= source.start
      && targetOffset <= source.end
    ) {
      draggedSelection.current = null;
      return;
    }
    let next = blocks.map((block) => ({ ...block }));
    const sourceBlock = next.find((block) => block.id === source.blockId);
    const targetBlock = next.find((block) => block.id === targetBlockId);
    if (!sourceBlock || !targetBlock) return;

    sourceBlock.text = sourceBlock.text.slice(0, source.start) + sourceBlock.text.slice(source.end);
    let insertionOffset = targetOffset;
    if (source.blockId === targetBlockId && insertionOffset > source.end) {
      insertionOffset -= source.end - source.start;
    }
    insertionOffset = Math.max(0, Math.min(targetBlock.text.length, insertionOffset));
    targetBlock.text = targetBlock.text.slice(0, insertionOffset) + source.text + targetBlock.text.slice(insertionOffset);
    draggedSelection.current = null;
    clearSelection();
    onChange(next, { kind: "command" });
  };

  const moveBlock = (sourceId: string, targetId: string) => {
    if (busy) return;
    if (sourceId === targetId) return;
    const sourceIndex = blocks.findIndex((block) => block.id === sourceId);
    const targetIndex = blocks.findIndex((block) => block.id === targetId);
    if (sourceIndex < 0 || targetIndex < 0) return;
    const next = [...blocks];
    const [moved] = next.splice(sourceIndex, 1);
    next.splice(targetIndex, 0, moved);
    onChange(next, { kind: "command" });
  };

  return (
    <section className="block-editor" aria-label="正文清洗编辑器">
      {selection && !busy && (
        <div className="selection-tools" style={{ left: selection.left, top: selection.top }}>
          <span>{selection.end - selection.start} 字</span>
          <button
            type="button"
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => replaceSelectedText("")}
          >删除</button>
          <button
            type="button"
            className="selection-drag-handle"
            draggable
            title="拖到任意段落中的位置"
            onDragStart={startSelectionDrag}
          >拖动</button>
          <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={clearSelection}>取消</button>
        </div>
      )}

      {blocks.map((block, visibleIndex) => (
        <article
          key={block.id}
          data-block-id={block.id}
          className={`editable-block ${block.deleted ? "is-dropped" : ""} ${activeBlockId === block.id ? "is-active" : ""}`}
          onPointerDown={() => onActiveBlockChange(block.id)}
          onDragOver={(event) => {
            if (event.dataTransfer.types.includes("application/x-curation-block")) {
              event.preventDefault();
              event.dataTransfer.dropEffect = "move";
            }
          }}
          onDrop={(event) => {
            if (!event.dataTransfer.types.includes("application/x-curation-block")) return;
            event.preventDefault();
            const sourceId = draggedBlockId.current;
            draggedBlockId.current = null;
            if (sourceId) moveBlock(sourceId, block.id);
          }}
        >
          <div className="editable-block-main">
          {block.deleted ? (
            <div className="editable-block-text deleted-preview">{block.text}</div>
          ) : (
            <EditableText
              block={block}
              disabled={busy}
              selection={selection}
              onActivate={() => onActiveBlockChange(block.id)}
              onTextChange={(text, kind) => onChange(
                blocks.map((value) => value.id === block.id ? { ...value, text } : value),
                kind === "typing" ? { kind, blockId: block.id } : { kind },
              )}
              onClearSelection={clearSelection}
              onSelectionConsumed={consumeSelection}
              onSelection={captureSelection}
              onSelectionDragStart={(event) => {
                if (selection?.blockId !== block.id) {
                  event.preventDefault();
                  return;
                }
                startSelectionDrag(event);
              }}
              onSelectionDrop={(_, targetOffset) => dropSelection(block.id, targetOffset)}
            />
          )}
          </div>
          <aside className="editable-block-meta" aria-label={`第 ${visibleIndex + 1} 段信息`}>
            <button
              type="button"
              className="block-drag-handle"
              draggable={!busy}
              title="拖动调整段落顺序"
              onDragStart={(event) => {
                draggedBlockId.current = block.id;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("application/x-curation-block", block.id);
              }}
            >⠿</button>
            <span>第 {visibleIndex + 1} 段</span>
            <span>{block.text.length.toLocaleString()} 字</span>
          </aside>
          <button
            type="button"
            className={`block-full-action ${block.deleted ? "restore-block" : "delete-block"}`}
            disabled={busy}
            onClick={() => {
              clearSelection();
              onChange(
                blocks.map((value) => value.id === block.id ? { ...value, deleted: !value.deleted } : value),
                { kind: "command" },
              );
            }}
          >{block.deleted ? "恢复" : "删除"}</button>
        </article>
      ))}
    </section>
  );
}
