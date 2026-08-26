from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _length_prefixed_digest(parts: tuple[str, ...], digest_size: int) -> str:
    digest = hashlib.blake2b(digest_size=digest_size)
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_source_id(source_id: str) -> str:
    if not isinstance(source_id, str) or not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ValueError(
            "source_id must start with an ASCII letter/digit and contain only "
            "letters, digits, '.', '_' or '-' (maximum 64 characters; this also "
            "keeps managed snapshot paths safe on Windows)"
        )
    return source_id


def sha256_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_source_revision(
    *,
    source_id: str,
    adapter_name: str,
    adapter_version: str,
    input_fingerprint: str,
    mapping: Mapping[str, Any],
    schema_version: str,
) -> str:
    validate_source_id(source_id)
    return _length_prefixed_digest(
        (
            source_id,
            adapter_name,
            adapter_version,
            input_fingerprint,
            stable_json(mapping),
            schema_version,
        ),
        digest_size=16,
    )


def make_doc_id(
    source_id: str,
    source_revision: str,
    stable_source_locator: str,
) -> str:
    validate_source_id(source_id)
    if not source_revision:
        raise ValueError("source_revision cannot be empty")
    if not stable_source_locator:
        raise ValueError("stable_source_locator cannot be empty")
    return _length_prefixed_digest(
        (source_id, source_revision, stable_source_locator),
        digest_size=16,
    )


def make_block_id(
    *,
    doc_id: str,
    parser_version: str,
    start_cp: int,
    end_cp: int,
    content_sha256: str,
) -> str:
    if start_cp < 0 or end_cp <= start_cp:
        raise ValueError("block offsets must satisfy 0 <= start_cp < end_cp")
    return _length_prefixed_digest(
        (
            doc_id,
            parser_version,
            str(start_cp),
            str(end_cp),
            content_sha256,
        ),
        digest_size=16,
    )
