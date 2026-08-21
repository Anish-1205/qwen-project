"""Central paths for local runtime state."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DATA_DIR = _env_path("CHATBOT_DATA_DIR", PROJECT_ROOT / "data")
AGENT_MEMORY_DB_PATH = _env_path("CHATBOT_MEMORY_DB", DATA_DIR / "agent_memory.db")
CHAT_SESSIONS_DB_PATH = _env_path("CHATBOT_SESSIONS_DB", DATA_DIR / "chat_sessions.db")
DOCUMENT_DB_PATH = _env_path("CHATBOT_DOCUMENT_DB", DATA_DIR / "documents.db")
DEBUG_LOG_PATH = _env_path("CHATBOT_DEBUG_LOG", DATA_DIR / "chatbot_debug.log")
