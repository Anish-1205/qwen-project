"""Environment-backed configuration for document discovery and retrieval."""

from __future__ import annotations

import os
from pathlib import Path

from app_paths import DOCUMENT_DB_PATH, PROJECT_ROOT

ROOT_DIR = PROJECT_ROOT


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


DOCS_DIR = _env_path("CHATBOT_DOCS_DIR", ROOT_DIR / "knowledge")
CHUNK_SIZE = _env_int("CHATBOT_DOC_CHUNK_SIZE", 350)
CHUNK_OVERLAP = _env_int("CHATBOT_DOC_CHUNK_OVERLAP", 60)
TOP_K = _env_int("CHATBOT_DOC_TOP_K", 4)
SIMILARITY_THRESHOLD = _env_float("CHATBOT_DOC_SIMILARITY_THRESHOLD", 0.55)
ROUTER_SIMILARITY_THRESHOLD = _env_float("CHATBOT_DOC_ROUTER_THRESHOLD", 0.22)
ROUTER_KEYWORD_BOOST = _env_float("CHATBOT_DOC_ROUTER_BOOST", 0.08)
MIN_PDF_CHARS_PER_PAGE = _env_int("CHATBOT_DOC_MIN_PDF_CHARS_PER_PAGE", 30)
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
ROUTER_KEYWORDS = (
    "manual",
    "policy",
    "document",
    "documents",
    "according to",
    "file",
    "files",
    "pdf",
    "txt",
    "guide",
    "spec",
    "specification",
    "notes",
    "readme",
    "report",
    "instructions",
    "csv",
    "docx",
    "markdown",
)
