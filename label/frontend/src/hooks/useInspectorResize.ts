import { useEffect, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent, PointerEvent } from "react";

const STORAGE_KEY = "label.inspector.width";
const MIN_WIDTH = 280;
const DEFAULT_WIDTH = 400;
const maxWidth = () => Math.max(MIN_WIDTH, window.innerWidth - 230 - 460 - 10);
const clampWidth = (width: number) => Math.round(Math.max(MIN_WIDTH, Math.min(maxWidth(), width)));

export function useInspectorResize() {
  const [width, setWidth] = useState(() => {
    try {
      const saved = Number(window.localStorage.getItem(STORAGE_KEY));
      return clampWidth(saved > 0 && Number.isFinite(saved) ? saved : DEFAULT_WIDTH);
    } catch { return clampWidth(DEFAULT_WIDTH); }
  });
  const [resizing, setResizing] = useState(false);
  const drag = useRef<{ x: number; width: number } | null>(null);
  useEffect(() => {
    try { window.localStorage.setItem(STORAGE_KEY, String(width)); } catch { /* Storage may be disabled. */ }
  }, [width]);
  useEffect(() => {
    const resize = () => setWidth((current) => clampWidth(current));
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  const endDrag = () => { drag.current = null; setResizing(false); };
  return {
    resizing,
    style: { "--inspector-width": `${width}px` } as CSSProperties,
    separatorProps: {
      role: "separator", tabIndex: 0, "aria-orientation": "vertical" as const,
      "aria-label": "调整右侧栏宽度", "aria-valuemin": MIN_WIDTH,
      "aria-valuemax": maxWidth(), "aria-valuenow": width,
      title: "拖动调整右侧栏宽度；双击恢复默认宽度",
      onDoubleClick: () => setWidth(clampWidth(DEFAULT_WIDTH)),
      onPointerDown: (event: PointerEvent<HTMLDivElement>) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.focus();
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = { x: event.clientX, width };
        setResizing(true);
      },
      onPointerMove: (event: PointerEvent<HTMLDivElement>) => {
        if (drag.current) setWidth(clampWidth(drag.current.width + drag.current.x - event.clientX));
      },
      onPointerUp: endDrag, onPointerCancel: endDrag, onLostPointerCapture: endDrag,
      onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        event.stopPropagation();
        setWidth(clampWidth(event.key === "Home" ? MIN_WIDTH : event.key === "End" ? maxWidth()
          : width + (event.key === "ArrowLeft" ? 20 : -20)));
      },
    },
  };
}
