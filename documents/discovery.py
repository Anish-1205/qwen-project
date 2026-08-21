from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS


def _normalized_path_key(path: Path) -> str:
    return path.resolve().as_posix().lower()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class DiscoveredFile:
    filepath: Path
    filepath_key: str
    document_id: str
    filename: str
    file_type: str
    file_size: int
    content_hash: str
    modified_at: float


@dataclass(slots=True)
class DiscoveryPlan:
    new: list[DiscoveredFile] = field(default_factory=list)
    changed: list[DiscoveredFile] = field(default_factory=list)
    unchanged: list[DiscoveredFile] = field(default_factory=list)
    deleted: list[dict] = field(default_factory=list)
    missing_directory: bool = False


def document_id_for_path(path: Path) -> str:
    return hashlib.sha256(_normalized_path_key(path).encode("utf-8")).hexdigest()


def scan_documents(docs_dir: Path) -> list[DiscoveredFile]:
    discovered: list[DiscoveredFile] = []
    if not docs_dir.exists():
        return discovered

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        file_type = path.suffix.lower()
        if file_type not in SUPPORTED_EXTENSIONS:
            continue

        stat_result = path.stat()
        discovered.append(
            DiscoveredFile(
                filepath=path,
                filepath_key=_normalized_path_key(path),
                document_id=document_id_for_path(path),
                filename=path.name,
                file_type=file_type,
                file_size=stat_result.st_size,
                content_hash=_hash_file(path),
                modified_at=stat_result.st_mtime,
            )
        )

    return discovered


def classify_documents(discovered: list[DiscoveredFile], existing_rows: dict[str, dict], missing_directory: bool = False) -> DiscoveryPlan:
    plan = DiscoveryPlan(missing_directory=missing_directory)
    discovered_keys = {item.filepath_key for item in discovered}

    for item in discovered:
        existing = existing_rows.get(item.filepath_key)
        if existing is None or existing.get("status") == "deleted":
            plan.new.append(item)
        elif existing.get("content_hash") == item.content_hash:
            plan.unchanged.append(item)
        else:
            plan.changed.append(item)

    for filepath_key, existing in existing_rows.items():
        if filepath_key not in discovered_keys and existing.get("status") != "deleted":
            plan.deleted.append(existing)

    return plan
