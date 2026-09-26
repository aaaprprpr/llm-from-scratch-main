from __future__ import annotations

from .base import SourceAdapter, SourceSpec
from .huggingface import HuggingFaceLocalAdapter
from .jsonl import JsonlAdapter
from .text import TextAdapter


ADAPTERS: dict[str, SourceAdapter] = {
    adapter.adapter_name: adapter
    for adapter in (
        HuggingFaceLocalAdapter(),
        JsonlAdapter(),
        TextAdapter(),
    )
}


def get_adapter(name: str) -> SourceAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown source adapter {name!r}; available={sorted(ADAPTERS)}"
        ) from exc


__all__ = ["ADAPTERS", "SourceAdapter", "SourceSpec", "get_adapter"]
