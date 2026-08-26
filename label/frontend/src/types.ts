export type Project = {
  project_id: string;
  name: string;
  guideline_version: string;
  current_revision: number;
};

export type Queue = {
  queue_id: string;
  project_id: string;
  name: string;
  sampling_seed: number;
  sampling_policy: Record<string, unknown>;
  state_counts: Record<string, number>;
  source_counts: Record<string, number>;
};

export type ProjectDetail = {
  project: Project;
  sources: Array<Record<string, unknown>>;
  queues: Queue[];
};

export type DocumentReview = {
  decision: "keep" | "drop" | "unsure";
  quality: number | null;
  primary_category: string | null;
  flags: string[];
  notes: string;
  revision: number;
};

export type BlockReview = {
  decision: "keep" | "drop";
  reason: string | null;
  revision: number;
};

export type ReviewBlock = {
  block_id: string;
  doc_id: string;
  ordinal: number;
  start_cp: number;
  end_cp: number;
  content_sha256: string;
  text: string;
  review: BlockReview | null;
};

export type QueueDocument = {
  queue: Queue;
  item: {
    project_id: string;
    queue_id: string;
    ordinal: number;
    state: "pending" | "done" | "skipped";
  };
  document: {
    doc_id: string;
    source_row: number;
    content_sha256: string;
  };
  raw_text: string;
  review_text: string;
  materialized_text: string;
  token_counts: {
    raw: number;
    review: number;
    materialized: number;
  } | null;
  tokenizer_path?: string;
  blocks: ReviewBlock[];
  document_review: DocumentReview | null;
  provenance: {
    source_id: string;
    source_revision: string;
    source_row: number;
    source_local_id: string | null;
    title: string | null;
    url: string | null;
    license: string;
    original_location: string;
  };
};
