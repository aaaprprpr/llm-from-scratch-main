from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .database import CurationDatabase, QueueCandidate
from .dataset_store import load_dataset


@dataclass(frozen=True)
class OpenProjectSource:
    source_id: str
    source_revision: str
    dataset_path: Path
    row_count: int
    dataset: Any

    @property
    def key(self) -> str:
        return f"{self.source_id}@{self.source_revision}"


def open_project_sources(
    database: CurationDatabase,
    project_id: str,
) -> list[OpenProjectSource]:
    attached = database.list_project_sources(project_id)
    if not attached:
        raise ValueError(f"project {project_id} has no attached sources")
    sources = []
    for value in attached:
        path = Path(value["dataset_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        dataset = load_dataset(path)
        row_count = int(value["row_count"])
        if len(dataset) != row_count:
            raise ValueError(
                f"attached source row count changed for {value['source_id']}: "
                f"database={row_count}, dataset={len(dataset)}"
            )
        sources.append(
            OpenProjectSource(
                source_id=value["source_id"],
                source_revision=value["source_revision"],
                dataset_path=path,
                row_count=row_count,
                dataset=dataset,
            )
        )
    return sources


def _candidate(source: OpenProjectSource, dataset_row: int) -> QueueCandidate:
    row = source.dataset[dataset_row]
    if row["source_id"] != source.source_id:
        raise ValueError("attached Dataset source_id does not match database")
    if row["source_revision"] != source.source_revision:
        raise ValueError("attached Dataset source_revision does not match database")
    if int(row["source_row"]) != dataset_row:
        raise ValueError(
            "V0.1 queues require source_row to equal the physical review row"
        )
    return QueueCandidate(
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_row=dataset_row,
        doc_id=row["doc_id"],
    )


def uniform_random_candidates(
    sources: Iterable[OpenProjectSource],
    *,
    sample_size: int,
    seed: int,
) -> list[QueueCandidate]:
    sources = list(sources)
    total = sum(source.row_count for source in sources)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > total:
        raise ValueError(f"sample_size {sample_size} exceeds {total} available rows")
    cumulative = []
    running_total = 0
    for source in sources:
        running_total += source.row_count
        cumulative.append(running_total)

    selected = random.Random(seed).sample(range(total), sample_size)
    candidates = []
    for global_row in selected:
        source_index = bisect.bisect_right(cumulative, global_row)
        source_start = 0 if source_index == 0 else cumulative[source_index - 1]
        candidates.append(
            _candidate(sources[source_index], global_row - source_start)
        )
    return candidates


def source_quota_candidates(
    sources: Iterable[OpenProjectSource],
    *,
    quotas: Mapping[str, int],
    seed: int,
) -> list[QueueCandidate]:
    sources = list(sources)
    by_key = {source.key: source for source in sources}
    by_id: dict[str, list[OpenProjectSource]] = {}
    for source in sources:
        by_id.setdefault(source.source_id, []).append(source)

    resolved: list[tuple[OpenProjectSource, int]] = []
    for requested_key, quota in quotas.items():
        if not isinstance(quota, int) or quota <= 0:
            raise ValueError(f"quota for {requested_key!r} must be positive")
        source = by_key.get(requested_key)
        if source is None:
            matches = by_id.get(requested_key, [])
            if len(matches) > 1:
                raise ValueError(
                    f"source_id {requested_key!r} has multiple revisions; use "
                    "source_id@source_revision"
                )
            if matches:
                source = matches[0]
        if source is None:
            raise KeyError(f"quota references unattached source {requested_key!r}")
        if quota > source.row_count:
            raise ValueError(
                f"quota {quota} exceeds {source.row_count} rows for {source.key}"
            )
        resolved.append((source, quota))

    if not resolved:
        raise ValueError("quotas cannot be empty")
    candidates = []
    rng = random.Random(seed)
    for source, quota in sorted(resolved, key=lambda value: value[0].key):
        source_seed = rng.getrandbits(128)
        indices = random.Random(source_seed).sample(range(source.row_count), quota)
        candidates.extend(_candidate(source, index) for index in indices)
    rng.shuffle(candidates)
    return candidates


def create_uniform_random_queue(
    database: CurationDatabase,
    *,
    project_id: str,
    name: str,
    sample_size: int,
    seed: int,
    queue_id: str | None = None,
) -> str:
    sources = open_project_sources(database, project_id)
    candidates = uniform_random_candidates(
        sources,
        sample_size=sample_size,
        seed=seed,
    )
    return database.create_queue(
        project_id=project_id,
        name=name,
        sampling_policy={"type": "uniform_random", "sample_size": sample_size},
        sampling_seed=seed,
        candidates=candidates,
        queue_id=queue_id,
    )


def create_source_quota_queue(
    database: CurationDatabase,
    *,
    project_id: str,
    name: str,
    quotas: Mapping[str, int],
    seed: int,
    queue_id: str | None = None,
) -> str:
    sources = open_project_sources(database, project_id)
    candidates = source_quota_candidates(sources, quotas=quotas, seed=seed)
    return database.create_queue(
        project_id=project_id,
        name=name,
        sampling_policy={"type": "source_quota", "quotas": dict(quotas)},
        sampling_seed=seed,
        candidates=candidates,
        queue_id=queue_id,
    )
