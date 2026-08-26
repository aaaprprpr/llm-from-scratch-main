from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Collection

from .identity import make_block_id, sha256_text


BLOCK_PARSER_VERSION = "blank_line_v1"
_BLOCK_SEPARATOR = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")


@dataclass(frozen=True)
class Block:
    block_id: str
    doc_id: str
    parser_version: str
    ordinal: int
    start_cp: int
    end_cp: int
    content_sha256: str
    text: str
    separator_after: str


def parse_blocks(
    text: str,
    doc_id: str,
    *,
    parser_version: str = BLOCK_PARSER_VERSION,
) -> list[Block]:
    if not isinstance(text, str) or not text:
        raise ValueError("block parser requires nonempty text")
    if text != text.strip():
        raise ValueError("review text must be stripped before block parsing")
    if parser_version != BLOCK_PARSER_VERSION:
        raise ValueError(f"Unsupported block parser version: {parser_version}")

    blocks: list[Block] = []
    cursor = 0
    for match in _BLOCK_SEPARATOR.finditer(text):
        content = text[cursor : match.start()]
        if content:
            content_hash = sha256_text(content)
            blocks.append(
                Block(
                    block_id=make_block_id(
                        doc_id=doc_id,
                        parser_version=parser_version,
                        start_cp=cursor,
                        end_cp=match.start(),
                        content_sha256=content_hash,
                    ),
                    doc_id=doc_id,
                    parser_version=parser_version,
                    ordinal=len(blocks),
                    start_cp=cursor,
                    end_cp=match.start(),
                    content_sha256=content_hash,
                    text=content,
                    separator_after=match.group(0),
                )
            )
        cursor = match.end()

    content = text[cursor:]
    if content:
        content_hash = sha256_text(content)
        blocks.append(
            Block(
                block_id=make_block_id(
                    doc_id=doc_id,
                    parser_version=parser_version,
                    start_cp=cursor,
                    end_cp=len(text),
                    content_sha256=content_hash,
                ),
                doc_id=doc_id,
                parser_version=parser_version,
                ordinal=len(blocks),
                start_cp=cursor,
                end_cp=len(text),
                content_sha256=content_hash,
                text=content,
                separator_after="",
            )
        )

    if not blocks:
        raise ValueError("block parser produced no blocks")
    return blocks


def render_blocks(
    original_text: str,
    blocks: list[Block],
    dropped_block_ids: Collection[str] = (),
) -> str:
    dropped = set(dropped_block_ids)
    known = {block.block_id for block in blocks}
    unknown = dropped - known
    if unknown:
        raise ValueError(f"Unknown dropped block ids: {sorted(unknown)}")

    if not dropped:
        reconstructed = "".join(
            block.text + block.separator_after for block in blocks
        )
        if reconstructed != original_text:
            raise RuntimeError("block parser failed lossless reconstruction")
        return reconstructed

    kept = [block.text for block in blocks if block.block_id not in dropped]
    return "\n\n".join(kept).strip()
