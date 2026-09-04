from __future__ import annotations

import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .common import ToolError, resolve_local_path


def list_directory(path: str, recursive: bool = False, max_depth: int = 1,
                   extensions: list[str] | None = None, pattern: str | None = None) -> dict:
    root = resolve_local_path(path)
    if not root.exists():
        raise ToolError("path_not_found", "The requested directory does not exist.", {"path": str(root)})
    if not root.is_dir():
        raise ToolError("not_a_directory", "The requested path is not a directory.", {"path": str(root)})
    depth_limit = min(max_depth, config.DIRECTORY_MAX_DEPTH) if recursive else 1
    wanted = {item.lower() if item.startswith(".") else f".{item.lower()}" for item in (extensions or [])}
    entries: list[dict] = []
    truncated = False

    def visit(directory: Path, depth: int) -> None:
        nonlocal truncated
        if truncated:
            return
        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
        except OSError as exc:
            raise ToolError("directory_read_error", "The directory could not be read.", {"path": str(directory)}) from exc
        for child in children:
            if child.is_symlink():
                continue
            is_dir = child.is_dir()
            matches = (not wanted or (not is_dir and child.suffix.lower() in wanted)) and (
                not pattern or fnmatch.fnmatch(child.name, pattern)
            )
            if matches:
                if len(entries) >= config.DIRECTORY_MAX_ENTRIES:
                    truncated = True
                    return
                try:
                    stat = child.stat()
                    modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
                    size = stat.st_size if child.is_file() else None
                except OSError:
                    modified, size = None, None
                entries.append({"name": child.name, "path": str(child.resolve()), "type": "directory" if is_dir else "file",
                                "size": size, "modified": modified})
                if len(json.dumps(entries, default=str)) > config.MAX_RESULT_CHARS:
                    entries.pop()
                    truncated = True
                    return
            if recursive and is_dir and depth < depth_limit:
                visit(child, depth + 1)

    visit(root, 0)
    return {"path": str(root), "entries": entries, "count": len(entries), "recursive": recursive,
            "max_depth": depth_limit, "truncated": truncated}
