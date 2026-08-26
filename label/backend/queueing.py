from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .database import CurationDatabase
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


def create_full_dataset_queue(
    database: CurationDatabase,
    *,
    project_id: str,
    name: str = "全量人工清洗",
    queue_id: str | None = None,
) -> str:
    attached = database.list_project_sources(project_id)
    if not attached:
        raise ValueError(f"project {project_id} has no attached sources")
    sources = [
        {
            "source_id": value["source_id"],
            "source_revision": value["source_revision"],
            "row_count": int(value["row_count"]),
        }
        for value in attached
    ]
    return database.create_virtual_queue(
        project_id=project_id,
        name=name,
        policy={"type": "full_dataset", "sources": sources},
        queue_id=queue_id,
    )
