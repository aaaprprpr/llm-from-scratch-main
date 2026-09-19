from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .schema import FieldMapping


class ProjectCreateRequest(BaseModel):
    name: str
    guideline_version: str = "1"
    project_id: str | None = None


class SourceRequest(BaseModel):
    adapter: str
    path: str
    options: dict = Field(default_factory=dict)


class MappingRequest(BaseModel):
    text_fields: list[str] = Field(default_factory=list)
    record_adapter: str | None = None
    text_separator: str = "\n\n"
    title_field: str | None = None
    url_field: str | None = None
    local_id_field: str | None = None
    metadata_fields: list[str] = Field(default_factory=list)

    def to_core(self) -> FieldMapping:
        return FieldMapping(
            text_fields=tuple(self.text_fields),
            text_separator=self.text_separator,
            title_field=self.title_field,
            url_field=self.url_field,
            local_id_field=self.local_id_field,
            metadata_fields=tuple(self.metadata_fields),
            record_adapter=self.record_adapter,
        )


class InspectRequest(SourceRequest):
    pass


class PreviewRequest(SourceRequest):
    mapping: MappingRequest
    limit: int = Field(default=20, ge=1, le=200)


class ImportRequest(SourceRequest):
    source_id: str | None = None
    license: str | None = None
    mapping: MappingRequest
    max_shard_size: str = "1GB"


class PrepareRequest(BaseModel):
    source_revision_directory: str
    config: dict = Field(default_factory=dict)
    max_shard_size: str = "1GB"
    read_batch_size: int = Field(default=1024, ge=1)


class AttachSourceRequest(BaseModel):
    prepare_revision_directory: str


class QueueCreateRequest(BaseModel):
    name: str
    policy: Literal["full_dataset"] = "full_dataset"
    queue_id: str | None = None


class SimplifyTextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=10_000)


class DocumentReviewRequest(BaseModel):
    queue_id: str
    ordinal: int = Field(ge=0)
    expected_revision: int = Field(ge=0)
    decision: Literal["keep", "drop", "unsure"]
    quality: int | None = Field(default=None, ge=0, le=3)
    primary_category: str | None = None
    flags: list[str] = Field(default_factory=list)
    notes: str = ""
    edited_text: str | None = None
    actor: str = "local-web"


class CleaningBlock(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(max_length=100000)
    separator_after: str = Field(default="\n\n", max_length=100000)


class LlmCleanRequest(BaseModel):
    queue_id: str
    ordinal: int = Field(ge=0)
    expected_revision: int = Field(ge=0)
    content_sha256: str
    blocks: list[CleaningBlock] = Field(min_length=1, max_length=10000)


class BlockReviewRequest(BaseModel):
    queue_id: str
    ordinal: int = Field(ge=0)
    expected_revision: int = Field(ge=0)
    decision: Literal["keep", "drop"]
    reason: str | None = None
    actor: str = "local-web"


class UndoRequest(BaseModel):
    project_id: str
    actor: str = "local-web"


class MaterializeRequest(BaseModel):
    project_id: str
    policy: Literal["keep_only", "drop_rejected"]
    snapshot_event_seq: int | None = Field(default=None, ge=0)
    max_shard_size: str = "1GB"
    read_batch_size: int = Field(default=1024, ge=1)


class PathPickerRequest(BaseModel):
    kind: Literal["directory", "file"]
    adapter: str = "huggingface_local"
    initial_directory: str | None = None
