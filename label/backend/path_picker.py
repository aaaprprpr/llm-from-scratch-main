from __future__ import annotations

from pathlib import Path
from typing import Literal


def select_local_path(
    *,
    kind: Literal["directory", "file"],
    adapter: str,
    initial_directory: str | Path | None = None,
) -> str | None:
    """Open a native picker on the machine running the label server."""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "This Python installation has no Tk support, so the native path "
            "picker is unavailable. Install python3-tk or use the manual path "
            "fallback."
        ) from exc

    initial = Path(initial_directory or Path.cwd()).resolve()
    if not initial.is_dir():
        initial = initial.parent

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        if kind == "directory":
            selected = filedialog.askdirectory(
                parent=root,
                title="选择数据集目录",
                initialdir=str(initial),
                mustexist=True,
            )
        else:
            file_types = {
                "jsonl": [("JSON Lines", "*.jsonl"), ("All files", "*.*")],
                "text": [("Text files", "*.txt"), ("All files", "*.*")],
            }.get(adapter, [("All files", "*.*")])
            selected = filedialog.askopenfilename(
                parent=root,
                title="选择数据文件",
                initialdir=str(initial),
                filetypes=file_types,
            )
        return str(Path(selected).resolve()) if selected else None
    except tk.TclError as exc:
        raise RuntimeError(
            "The native path picker could not open a desktop window. If the "
            "server is headless, use the manual path fallback."
        ) from exc
    finally:
        if root is not None:
            root.destroy()
