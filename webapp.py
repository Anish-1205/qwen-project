"""Serve the local FastAPI chat UI, session API, and document management API.

The server is designed for a single trusted user on localhost. It deliberately
does not provide authentication or internet-facing deployment hardening.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").disabled = True
warnings.filterwarnings("ignore", message="triton not found")

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from huggingface_hub.utils import logging as hf_logging
from markdown_it import MarkdownIt
import nh3
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app_paths import CHAT_SESSIONS_DB_PATH
from documents import DocumentIndex
from documents.db import open_db as open_documents_db
from logging_utils import capture_prints, launch_log_tailer, setup_debug_logger
from memory_core import OfflineMemoryManager
from orchestrator import ConversationOrchestrator, DEFAULT_SYSTEM_PROMPT, strip_speaker_tags

hf_logging.set_verbosity_error()

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SESSION_COOKIE = "roots_chat_session"
APP_TITLE = "Roots Chat"
DEFAULT_SESSION_TITLE = "New Chat"
SESSION_TITLE_MAX_LENGTH = 48
DOCUMENT_UPLOAD_MAX_BYTES = 10_000_000
DOCUMENT_UPLOAD_EXTENSIONS = {".pdf", ".txt"}

MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
MARKDOWN_ALLOWED_TAGS = {
  "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4", "h5", "h6",
  "hr", "li", "ol", "p", "pre", "strong", "ul",
}
MARKDOWN_ALLOWED_ATTRIBUTES = {"a": {"href", "title"}, "code": {"class"}}
MARKDOWN_ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}


class ChatRequest(BaseModel):
  message: str = Field(min_length=1, max_length=20000)
  session_id: str | None = None


@dataclass
class SessionState:
  session_id: str
  title: str = DEFAULT_SESSION_TITLE
  messages: list[dict] = field(default_factory=lambda: [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}])
  turn_number: int = 0


@dataclass
class AppState:
  status: str = "initializing"
  detail: str = "Starting up."
  error: str | None = None
  logger: logging.Logger | None = None
  log_path: Path | None = None
  tokenizer: object | None = None
  model: object | None = None
  memory: OfflineMemoryManager | None = None
  document_index: DocumentIndex | None = None
  orchestrator: ConversationOrchestrator | None = None
  init_started: bool = False
  init_lock: threading.Lock = field(default_factory=threading.Lock)
  generation_lock: threading.Lock = field(default_factory=threading.Lock)
  session_store_lock: threading.Lock = field(default_factory=threading.Lock)
  document_sync_lock: threading.Lock = field(default_factory=threading.Lock)
  sessions: dict[str, SessionState] = field(default_factory=dict)


class ReadOnlyOfflineMemoryManager(OfflineMemoryManager):
  """Use the shared retrieval implementation without LRU or promotion writes."""

  retrieval_mutates_cache = False


APP_STATE = AppState()
app = FastAPI(title=APP_TITLE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_session_title(title: str | None) -> str:
    cleaned = " ".join((title or DEFAULT_SESSION_TITLE).strip().split())
    if not cleaned:
        return DEFAULT_SESSION_TITLE
    if len(cleaned) <= SESSION_TITLE_MAX_LENGTH:
        return cleaned
    return cleaned[: SESSION_TITLE_MAX_LENGTH - 1].rstrip() + "…"


def _title_from_message(message: str) -> str:
    normalized = " ".join((message or "").strip().split())
    if not normalized:
        return DEFAULT_SESSION_TITLE
    if len(normalized) <= SESSION_TITLE_MAX_LENGTH:
        return normalized
    return normalized[: SESSION_TITLE_MAX_LENGTH - 1].rstrip() + "…"


def _chat_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(CHAT_SESSIONS_DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _init_chat_sessions_db() -> None:
    CHAT_SESSIONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _chat_db_connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_turn ON messages(session_id, turn_number, id)")


def _session_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _message_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "turn_number": row["turn_number"],
        "created_at": row["created_at"],
    }


def _render_assistant_markdown(content: str) -> str:
  """Render trusted Markdown syntax while keeping embedded HTML and unsafe URLs inert."""
  rendered = MARKDOWN_RENDERER.render(content or "")
  return nh3.clean(
    rendered,
    tags=MARKDOWN_ALLOWED_TAGS,
    attributes=MARKDOWN_ALLOWED_ATTRIBUTES,
    url_schemes=MARKDOWN_ALLOWED_URL_SCHEMES,
    link_rel="noopener noreferrer",
    strip_comments=True,
  )


def _message_to_response(message: dict) -> dict:
  response_message = dict(message)
  if response_message.get("role") == "assistant":
    response_message["rendered_content"] = _render_assistant_markdown(response_message.get("content", ""))
  return response_message


def _create_session_record(title: str = DEFAULT_SESSION_TITLE, session_id: str | None = None) -> dict:
    session_id = session_id or uuid.uuid4().hex
    now = _utc_now_iso()
    normalized_title = _normalize_session_title(title)
    with APP_STATE.session_store_lock, _chat_db_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, normalized_title, now, now),
        )
    with APP_STATE.session_store_lock:
        APP_STATE.sessions[session_id] = SessionState(session_id=session_id, title=normalized_title)
    return {"id": session_id, "title": normalized_title, "created_at": now, "updated_at": now}


def _session_exists(session_id: str) -> bool:
  with _chat_db_connection() as conn:
    row = conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone()
  return row is not None


def _load_session_state(session_id: str) -> SessionState | None:
  with APP_STATE.session_store_lock, _chat_db_connection() as conn:
    session_row = conn.execute(
      "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
      (session_id,),
    ).fetchone()
    if session_row is None:
      return None

    message_rows = conn.execute(
      "SELECT id, session_id, role, content, turn_number, created_at FROM messages WHERE session_id = ? ORDER BY turn_number, id",
      (session_id,),
    ).fetchall()

  messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
  turn_number = 0
  for message_row in message_rows:
    message = _message_row_to_dict(message_row)
    messages.append({"role": message["role"], "content": message["content"]})
    turn_number = max(turn_number, int(message["turn_number"]))

  state = SessionState(
    session_id=session_row["id"],
    title=session_row["title"],
    messages=messages,
    turn_number=turn_number,
  )
  with APP_STATE.session_store_lock:
    APP_STATE.sessions[session_id] = state
  return state


def _get_or_create_session_state(session_id: str, title: str = DEFAULT_SESSION_TITLE) -> SessionState:
  with APP_STATE.session_store_lock:
    cached_state = APP_STATE.sessions.get(session_id)
  if cached_state is not None:
    return cached_state

  loaded_state = _load_session_state(session_id)
  if loaded_state is not None:
    return loaded_state

  _create_session_record(title=title, session_id=session_id)
  with APP_STATE.session_store_lock:
    return APP_STATE.sessions[session_id]


def _list_session_records() -> list[dict]:
  with _chat_db_connection() as conn:
    rows = conn.execute(
      "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC, created_at DESC"
    ).fetchall()
  return [_session_row_to_dict(row) for row in rows]


def _get_session_record(session_id: str) -> dict | None:
  with _chat_db_connection() as conn:
    row = conn.execute(
      "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
      (session_id,),
    ).fetchone()
  return _session_row_to_dict(row) if row is not None else None


def _get_session_messages(session_id: str) -> list[dict]:
  with _chat_db_connection() as conn:
    rows = conn.execute(
      "SELECT id, session_id, role, content, turn_number, created_at FROM messages WHERE session_id = ? ORDER BY turn_number, id",
      (session_id,),
    ).fetchall()
  return [_message_row_to_dict(row) for row in rows]


def _persist_turn(session_id: str, turn_number: int, user_message: str, assistant_reply: str, title_source: str | None = None) -> None:
  now = _utc_now_iso()
  title = _normalize_session_title(title_source) if title_source else None
  with APP_STATE.session_store_lock, _chat_db_connection() as conn:
    if title is not None:
      conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, now, session_id),
      )
    else:
      conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.execute(
      "INSERT INTO messages (session_id, role, content, turn_number, created_at) VALUES (?, ?, ?, ?, ?)",
      (session_id, "user", user_message, turn_number, now),
    )
    conn.execute(
      "INSERT INTO messages (session_id, role, content, turn_number, created_at) VALUES (?, ?, ?, ?, ?)",
      (session_id, "assistant", assistant_reply, turn_number, now),
    )


def _delete_session_record(session_id: str) -> bool:
  with APP_STATE.session_store_lock, _chat_db_connection() as conn:
    deleted_messages = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,)).rowcount
    deleted_sessions = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount
  with APP_STATE.session_store_lock:
    APP_STATE.sessions.pop(session_id, None)
  return deleted_sessions > 0 or deleted_messages > 0


def _get_documents_root() -> Path:
  if APP_STATE.document_index is not None:
    return Path(APP_STATE.document_index.docs_dir)
  return Path(__file__).resolve().parent / "knowledge"


def _human_file_size(num_bytes: int) -> str:
  value = float(num_bytes)
  for unit in ["B", "KB", "MB", "GB"]:
    if value < 1024.0 or unit == "GB":
      if unit == "B":
        return f"{int(value)} {unit}"
      return f"{value:.1f} {unit}"
    value /= 1024.0
  return f"{int(value)} B"


def _list_documents() -> list[dict]:
  with open_documents_db() as conn:
    rows = conn.execute(
      """
      SELECT d.id, d.filename, d.filepath, d.file_type, d.file_size, d.created_at, d.modified_at, d.status,
             COALESCE(COUNT(c.id), 0) AS chunk_count
      FROM documents d
      LEFT JOIN chunks c ON c.document_id = d.id
      GROUP BY d.id, d.filename, d.filepath, d.file_type, d.file_size, d.created_at, d.modified_at, d.status
      ORDER BY d.modified_at DESC, d.filename ASC
      """
    ).fetchall()

  documents: list[dict] = []
  for row in rows:
    chunk_count = int(row["chunk_count"] or 0)
    status = row["status"]
    if status == "indexed":
      summary = f"{chunk_count} chunk{'s' if chunk_count != 1 else ''} indexed"
    elif status == "empty_extraction":
      summary = "empty_extraction"
    elif status == "failed":
      summary = "failed"
    elif status == "deleted":
      summary = "deleted"
    else:
      summary = status
    documents.append(
      {
        "id": row["id"],
        "filename": row["filename"],
        "filepath": row["filepath"],
        "file_type": row["file_type"],
        "file_size": int(row["file_size"] or 0),
        "file_size_label": _human_file_size(int(row["file_size"] or 0)),
        "created_at": row["created_at"],
        "modified_at": row["modified_at"],
        "status": status,
        "chunk_count": chunk_count,
        "summary": summary,
      }
    )
  return documents


def _sync_documents() -> dict:
  if APP_STATE.document_index is None:
    raise HTTPException(status_code=503, detail="Document index is not ready")
  with APP_STATE.document_sync_lock:
    plan = APP_STATE.document_index.sync()
  return {
    "new": len(plan.new),
    "changed": len(plan.changed),
    "unchanged": len(plan.unchanged),
    "deleted": len(plan.deleted),
    "missing_directory": plan.missing_directory,
  }


def _save_uploaded_document(filename: str, content: bytes) -> Path:
  docs_root = _get_documents_root()
  docs_root.mkdir(parents=True, exist_ok=True)
  safe_name = Path(filename).name
  if not safe_name or safe_name in {".", ".."}:
    raise HTTPException(status_code=400, detail="Invalid filename")
  if Path(safe_name).suffix.lower() not in DOCUMENT_UPLOAD_EXTENSIONS:
    raise HTTPException(status_code=415, detail="Only .txt and .pdf documents are supported")
  if len(content) > DOCUMENT_UPLOAD_MAX_BYTES:
    raise HTTPException(status_code=413, detail="Document exceeds the upload size limit")
  target = docs_root / safe_name
  try:
    with target.open("xb") as output:
      output.write(content)
  except FileExistsError as exc:
    raise HTTPException(status_code=409, detail="A document with that filename already exists") from exc
  return target


def _delete_document_file(document_id: str) -> Path:
  with open_documents_db() as conn:
    row = conn.execute("SELECT filepath FROM documents WHERE id = ?", (document_id,)).fetchone()
  if row is None:
    raise HTTPException(status_code=404, detail="Document not found")

  filepath = Path(row["filepath"])
  if filepath.exists():
    filepath.unlink()
  return filepath


def _session_to_response(session_id: str) -> dict:
  session_record = _get_session_record(session_id)
  if session_record is None:
    raise HTTPException(status_code=404, detail="Session not found")
  return session_record


def _session_payload(session_id: str) -> dict:
  session_record = _get_session_record(session_id)
  if session_record is None:
    raise HTTPException(status_code=404, detail="Session not found")
  return {
    "session": session_record,
    "messages": [_message_to_response(message) for message in _get_session_messages(session_id)],
  }


def _activate_browser_session(session_id: str, response: Response | None = None) -> dict:
  session_record = _get_session_record(session_id)
  if session_record is None:
    raise HTTPException(status_code=404, detail="Session not found")
  _get_or_create_session_state(session_id, title=session_record["title"])
  if response is not None:
    response.set_cookie(
      SESSION_COOKIE,
      session_id,
      httponly=True,
      samesite="lax",
      path="/",
    )
  return session_record


_init_chat_sessions_db()


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _set_state(**updates) -> None:
    with APP_STATE.init_lock:
        for key, value in updates.items():
            setattr(APP_STATE, key, value)


def _get_or_create_session_id(request: Request, response: Response) -> str:
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id and _session_exists(session_id):
        return session_id

    session_record = _create_session_record(DEFAULT_SESSION_TITLE)
    session_id = session_record["id"]
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return session_id


def _get_session(session_id: str) -> SessionState:
    return _get_or_create_session_state(session_id)


def _serialize_retrieval_metadata(metadata) -> dict:
    return {
        "memory": {
            "retrieved": metadata.memory.retrieved,
            "facts": list(metadata.memory.facts),
        },
        "documents": {
            "retrieved": metadata.documents.retrieved,
            "chunks": [
                {
                    "source": chunk.source,
                    "location": chunk.location,
                    "text": chunk.text,
                    "injected": chunk.injected,
                }
                for chunk in metadata.documents.chunks
            ],
        },
    }


def _serialize_document_result(document_result) -> dict:
    return {
        "context": document_result.context,
        "routed_relevant": document_result.routed_relevant,
        "retrieved_count": document_result.retrieved_count,
        "sources": list(document_result.sources),
        "reason": document_result.reason,
    }


def _initialize_app() -> None:
    with APP_STATE.init_lock:
        if APP_STATE.init_started:
            return
        APP_STATE.init_started = True
        APP_STATE.status = "initializing"
        APP_STATE.detail = "Loading model and document index."
        APP_STATE.error = None

    logger, log_path = setup_debug_logger()
    _set_state(logger=logger, log_path=log_path)

    if not launch_log_tailer(log_path, logger):
        logger.info("Debug log: %s", log_path)

    try:
        with capture_prints(logger):
            logger.info("Loading Qwen2.5-3B-Instruct model in 4-bit NF4...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=_build_bnb_config(),
                dtype="auto",
                device_map="auto",
            )
            model.eval()
            logger.info("Model loading complete.")

            memory = ReadOnlyOfflineMemoryManager()
            document_index = DocumentIndex(memory.embed_model, logger=logger)
            document_index.sync()

        orchestrator = ConversationOrchestrator(
            tokenizer,
            model,
            memory,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            compression_enabled=True,
            reply_generation_kwargs={
              "max_new_tokens": 450,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            router_generation_kwargs={
                "max_new_tokens": 120,
                "do_sample": False,
            },
            document_lookup=document_index.lookup_context,
            logger=logger.info,
        )

        _set_state(
            tokenizer=tokenizer,
            model=model,
            memory=memory,
            document_index=document_index,
            orchestrator=orchestrator,
            status="ready",
            detail="Ready.",
        )
        logger.info("Web app initialization complete.")
    except Exception as exc:  # pragma: no cover - startup failure path
        _set_state(status="failed", detail="Initialization failed.", error=str(exc))
        logger.exception("Web app initialization failed: %s", exc)


def _get_state_snapshot() -> dict:
    with APP_STATE.init_lock:
        return {
            "status": APP_STATE.status,
            "detail": APP_STATE.detail,
            "error": APP_STATE.error,
            "log_path": str(APP_STATE.log_path) if APP_STATE.log_path else None,
        }


def _format_trace_html(initial_session_id: str) -> str:
  html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Roots Chat</title>
  <style>
    :root {
      --bg: #070b14;
      --bg2: #0c1424;
      --panel: rgba(10, 16, 28, 0.84);
      --panel-soft: rgba(255, 255, 255, 0.04);
      --border: rgba(255, 255, 255, 0.10);
      --text: #edf3ff;
      --muted: #91a4c7;
      --accent: #7de3d0;
      --accent-2: #f5b971;
      --accent-3: #8dd3ff;
      --shadow: 0 24px 60px rgba(0, 0, 0, 0.38);
      --radius-xl: 28px;
      --radius-lg: 20px;
      --radius-md: 14px;
      --radius-sm: 10px;
      font-family: "Aptos", "Segoe UI Variable", "Segoe UI", "Helvetica Neue", sans-serif;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; overflow: hidden; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(125, 227, 208, 0.17), transparent 34%),
        radial-gradient(circle at top right, rgba(141, 211, 255, 0.16), transparent 32%),
        radial-gradient(circle at bottom left, rgba(245, 185, 113, 0.11), transparent 24%),
        linear-gradient(180deg, var(--bg), var(--bg2));
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image: linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.55), transparent 88%);
      opacity: 0.4;
    }

    .shell {
      position: relative;
      height: 100%;
      min-height: 100%;
      display: grid;
      grid-template-columns: 300px minmax(0, 1.55fr) minmax(280px, 0.95fr);
      gap: 20px;
      padding: 24px;
      max-width: 1480px;
      margin: 0 auto;
      overflow: hidden;
    }

    .shell.left-collapsed {
      grid-template-columns: minmax(0, 1fr) minmax(280px, 0.95fr);
    }

    .shell.right-collapsed {
      grid-template-columns: 300px minmax(0, 1fr);
    }

    .shell.left-collapsed.right-collapsed {
      grid-template-columns: minmax(0, 1fr);
      gap: 0;
      max-width: none;
      padding-left: 0;
      padding-right: 0;
    }

    .panel-rail {
      position: fixed;
      top: 18px;
      z-index: 40;
      pointer-events: none;
    }

    .panel-rail-left {
      left: 18px;
    }

    .panel-rail-right {
      right: 18px;
    }

    .rail-btn {
      pointer-events: auto;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(10, 16, 28, 0.86);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
      box-shadow: var(--shadow-soft);
      backdrop-filter: blur(14px);
    }

    .rail-btn[aria-expanded="false"] {
      color: var(--accent);
      border-color: rgba(125, 227, 208, 0.22);
    }

    .sessions, .main, .sidebar {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      min-width: 0;
      min-height: 0;
    }

    .sessions.is-collapsed, .sidebar.is-collapsed {
      display: none;
    }

    .sessions {
      display: flex;
      flex-direction: column;
      gap: 16px;
      min-height: 0;
      padding: 18px;
      overflow: hidden;
    }

    .sessions-header {
      display: grid;
      gap: 10px;
    }

    .panel-toolbar,
    .card-head,
    .modal-header,
    .modal-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-width: 0;
      flex-wrap: wrap;
    }

    .panel-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-left: auto;
      flex-wrap: wrap;
    }

    .panel-toolbar h2,
    .card-head h2,
    .modal-header h2 {
      min-width: 0;
      margin-right: auto;
    }

    .panel-toolbar {
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      align-items: flex-start;
    }

    .sessions-header h2 {
      margin: 0;
      font-size: 1.05rem;
      letter-spacing: -0.03em;
    }

    .new-chat-btn {
      width: 100%;
      border: none;
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      font-weight: 700;
      color: #07111a;
      background: linear-gradient(135deg, var(--accent), var(--accent-3));
      cursor: pointer;
    }

    .icon-btn,
    .upload-btn {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
      transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
    }

    .icon-btn:hover,
    .upload-btn:hover {
      transform: translateY(-1px);
      border-color: rgba(125, 227, 208, 0.22);
      background: rgba(255,255,255,0.07);
    }

    .icon-btn[aria-expanded="false"] {
      border-color: rgba(125, 227, 208, 0.22);
      color: var(--accent);
    }
    .main {
      display: flex;
      flex-direction: column;
      min-height: 0;
      min-width: 0;
      overflow: hidden;
    }

    .session-list {
      display: flex;
      flex-direction: column;
      flex: 1 1 auto;
      min-height: 0;
      gap: 2px;
      overflow-y: auto;
      overflow-x: hidden;
      padding-right: 4px;
    }

    .session-item {
      position: relative;
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      width: 100%;
      text-align: left;
      border: none;
      border-radius: 10px;
      padding: 8px 10px;
      background: transparent;
      color: var(--text);
      cursor: pointer;
      transition: background 120ms ease, color 120ms ease;
    }

    .session-item:hover {
      background: rgba(255,255,255,0.05);
    }

    .session-item.active {
      background: rgba(255,255,255,0.08);
    }

    .session-item .session-delete {
      opacity: 0;
      pointer-events: none;
      position: static;
      margin-left: auto;
      border: none;
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-size: 1rem;
      line-height: 1;
      cursor: pointer;
      padding: 0;
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      flex: 0 0 28px;
    }

    .session-item:hover .session-delete,
    .session-item:focus-within .session-delete {
      opacity: 1;
      pointer-events: auto;
    }

    .session-title {
      display: block;
      flex: 1 1 auto;
      min-width: 0;
      font-weight: 500;
      line-height: 1.35;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .session-meta {
      display: none;
    }

    .card-head {
      margin-bottom: 10px;
      align-items: center;
    }

    .card-head h2 {
      margin: 0;
    }

    .card-subtle {
      color: var(--muted);
      font-size: 0.82rem;
    }

    .kv {
      display: grid;
      gap: 10px;
      font-size: 0.92rem;
      color: var(--muted);
      line-height: 1.45;
    }

    .kv strong { color: var(--text); font-weight: 600; }

    .kv div,
    .trace-item,
    .chunk-text,
    .document-row,
    .modal-shell .muted {
      min-width: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }

    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 20px;
      align-content: start;
      min-height: 0;
      overflow: auto;
    }

    .card-head {
      margin-bottom: 10px;
    }

    .card-head h2 {
      margin: 0;
    }

    .card-subtle {
      color: var(--muted);
      font-size: 0.82rem;
    }

    .modal-backdrop {
      position: fixed;
      inset: 0;
      display: grid;
      place-items: center;
      background: rgba(4, 8, 16, 0.58);
      padding: 24px;
      z-index: 50;
    }

    .modal-backdrop[hidden] {
      display: none !important;
    }

    .modal-shell {
      width: min(980px, 100%);
      max-height: min(84vh, 900px);
      display: grid;
      gap: 16px;
      padding: 20px;
      border-radius: 24px;
      background: rgba(10, 16, 28, 0.96);
      border: 1px solid rgba(255,255,255,0.09);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .modal-toolbar {
      padding: 0 2px;
    }

    .upload-btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .document-list {
      display: grid;
      gap: 10px;
      overflow: auto;
      padding-right: 4px;
    }

    .document-row {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(120px, 0.7fr) minmax(130px, 0.8fr) auto;
      gap: 10px;
      align-items: center;
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
    }

    .document-name {
      font-weight: 700;
      word-break: break-word;
    }

    .document-meta {
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.4;
    }

    .document-status {
      color: var(--text);
      font-size: 0.9rem;
      line-height: 1.35;
    }

    .document-actions {
      display: flex;
      gap: 8px;
      justify-content: end;
      align-items: center;
    }

    .document-delete {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }

    .document-empty,
    .panel-empty {
      padding: 12px 0;
      color: var(--muted);
    }

    .busy-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(245, 185, 113, 0.10);
      border: 1px solid rgba(245, 185, 113, 0.24);
      color: var(--accent-2);
      font-size: 0.84rem;
    }

    .main {
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
    }

    .hero {
      padding: 26px 28px 18px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.05), transparent);
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.24em;
      margin-bottom: 12px;
    }

    .eyebrow .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 16px rgba(125, 227, 208, 0.85);
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.8rem);
      line-height: 0.98;
      letter-spacing: -0.06em;
      max-width: 10ch;
    }

    .subhead {
      margin-top: 14px;
      color: var(--muted);
      max-width: 62ch;
      line-height: 1.55;
      font-size: 0.98rem;
    }

    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 18px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      color: var(--muted);
      font-size: 0.92rem;
    }

    .status-chip .pulse {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--accent-2);
      box-shadow: 0 0 0 0 rgba(245, 185, 113, 0.55);
      animation: pulse 2.2s infinite;
    }

    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(245, 185, 113, 0.55); }
      70% { box-shadow: 0 0 0 12px rgba(245, 185, 113, 0); }
      100% { box-shadow: 0 0 0 0 rgba(245, 185, 113, 0); }
    }

    .messages {
      overflow: auto;
      padding: 24px 18px 8px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .message {
      display: grid;
      gap: 10px;
      padding: 18px 18px 16px;
      border-radius: var(--radius-lg);
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      max-width: min(100%, 920px);
      animation: rise 280ms ease-out;
    }

    .message.user {
      align-self: flex-end;
      background: linear-gradient(180deg, rgba(245, 185, 113, 0.12), rgba(245, 185, 113, 0.05));
      border-color: rgba(245, 185, 113, 0.22);
    }

    .message.assistant {
      align-self: flex-start;
      background: linear-gradient(180deg, rgba(125, 227, 208, 0.11), rgba(141, 211, 255, 0.05));
      border-color: rgba(125, 227, 208, 0.18);
    }

    .message.system {
      align-self: center;
      max-width: 68ch;
      color: var(--muted);
      font-size: 0.92rem;
    }

    @keyframes rise {
      from { transform: translateY(6px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }

    .role {
      font-size: 0.78rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .content {
      line-height: 1.65;
      font-size: 1rem;
      min-width: 0;
      text-align: left;
      overflow-wrap: break-word;
    }

    .message.user .content,
    .message.system .content {
      white-space: pre-wrap;
    }

    .message.assistant .content > :first-child { margin-top: 0; }
    .message.assistant .content > :last-child { margin-bottom: 0; }

    .message.assistant .content p {
      margin: 0 0 0.9em;
    }

    .message.assistant .content h1,
    .message.assistant .content h2,
    .message.assistant .content h3,
    .message.assistant .content h4,
    .message.assistant .content h5,
    .message.assistant .content h6 {
      margin: 1.25em 0 0.55em;
      color: var(--text);
      font-weight: 700;
      line-height: 1.25;
      letter-spacing: -0.02em;
    }

    .message.assistant .content h1 { font-size: 1.45rem; }
    .message.assistant .content h2 { font-size: 1.28rem; }
    .message.assistant .content h3 { font-size: 1.14rem; }
    .message.assistant .content h4,
    .message.assistant .content h5,
    .message.assistant .content h6 { font-size: 1rem; }

    .message.assistant .content ul,
    .message.assistant .content ol {
      margin: 0.55em 0 0.95em;
      padding-left: 1.55em;
    }

    .message.assistant .content li {
      margin: 0.28em 0;
      padding-left: 0.15em;
    }

    .message.assistant .content li > ul,
    .message.assistant .content li > ol {
      margin: 0.25em 0;
    }

    .message.assistant .content strong { font-weight: 700; color: var(--text); }

    .message.assistant .content a {
      color: var(--accent-3);
      text-decoration: underline;
      text-decoration-thickness: 0.08em;
      text-underline-offset: 0.16em;
      overflow-wrap: anywhere;
    }

    .message.assistant .content code {
      padding: 0.12em 0.35em;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.28);
      font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      font-size: 0.9em;
    }

    .message.assistant .content pre {
      max-width: 100%;
      margin: 0.75em 0 1em;
      padding: 14px 16px;
      overflow-x: auto;
      border: 1px solid rgba(255,255,255,0.09);
      border-radius: var(--radius-md);
      background: rgba(3, 7, 14, 0.72);
      line-height: 1.5;
    }

    .message.assistant .content pre code {
      padding: 0;
      background: transparent;
      white-space: pre;
      overflow-wrap: normal;
    }

    .message.assistant .content blockquote {
      margin: 0.75em 0 1em;
      padding: 0.15em 0 0.15em 1em;
      border-left: 3px solid rgba(125, 227, 208, 0.45);
      color: var(--muted);
    }

    .message.assistant .content hr {
      margin: 1.2em 0;
      border: 0;
      border-top: 1px solid rgba(255,255,255,0.12);
    }

    .composer {
      padding: 18px 20px 22px;
      border-top: 1px solid var(--border);
      background: linear-gradient(180deg, transparent, rgba(0,0,0,0.18));
    }

    .composer-panel {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
      padding: 14px;
      border-radius: calc(var(--radius-lg) + 4px);
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
    }

    textarea {
      width: 100%;
      height: 72px;
      min-height: 72px;
      max-height: min(42vh, 420px);
      resize: none;
      overflow-y: hidden;
      border: none;
      outline: none;
      color: var(--text);
      background: transparent;
      font: inherit;
      line-height: 1.55;
      padding: 2px 4px;
    }

    textarea::placeholder { color: rgba(145, 164, 199, 0.75); }

    .actions {
      display: flex;
      gap: 10px;
      align-items: center;
    }

    button {
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      letter-spacing: 0.02em;
      transition: transform 140ms ease, opacity 140ms ease, background 140ms ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

    .primary {
      color: #07111a;
      background: linear-gradient(135deg, var(--accent), var(--accent-3));
      box-shadow: 0 10px 26px rgba(125, 227, 208, 0.18);
    }

    .secondary {
      color: var(--text);
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
    }

    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 20px;
      align-content: start;
      min-height: 0;
      overflow: auto;
    }

    .card {
      padding: 16px;
      border-radius: var(--radius-lg);
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.035);
    }

    .card h2 {
      margin: 0 0 10px;
      font-size: 0.92rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
    }

    .state-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 0.88rem;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
    }

    .badge .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--accent-2);
    }

    .kv {
      display: grid;
      gap: 10px;
      font-size: 0.92rem;
      color: var(--muted);
      line-height: 1.45;
      min-width: 0;
    }

    .kv strong { color: var(--text); font-weight: 600; }

    .trace-list {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }

    .trace-item {
      padding: 12px;
      border-radius: var(--radius-md);
      background: rgba(0,0,0,0.16);
      border: 1px solid rgba(255,255,255,0.06);
      font-size: 0.9rem;
      color: var(--muted);
    }

    .trace-item .head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 8px;
      color: var(--text);
    }

    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.07);
      font-size: 0.82rem;
      color: var(--text);
    }

    .chunk-text {
      margin-top: 8px;
      white-space: pre-wrap;
      line-height: 1.5;
    }

    .muted {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.5;
    }

    .spinner {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .spinner::before {
      content: "";
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 2px solid rgba(255,255,255,0.16);
      border-top-color: var(--accent);
      animation: spin 800ms linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    @media (max-width: 1080px) {
      .shell { grid-template-columns: 1fr; }
      .sessions, .sidebar, .main { min-height: auto; }
      .main { min-height: 0; }
    }

    @media (max-width: 720px) {
      .shell { padding: 12px; gap: 12px; }
      .hero, .composer, .messages, .sidebar, .sessions { padding-left: 16px; padding-right: 16px; }
      .composer-panel { grid-template-columns: 1fr; }
      .actions { justify-content: flex-end; }
    }
  </style>
</head>
<body>
  <script>
    window.__INITIAL_SESSION_ID__ = "__INITIAL_SESSION_ID_PLACEHOLDER__";
  </script>
  <div class="shell">
    <div class="panel-rail panel-rail-left">
      <button class="rail-btn" id="open-left-btn" type="button" aria-expanded="true">Chats</button>
    </div>
    <aside class="sessions">
      <div class="sessions-header">
        <div class="panel-toolbar">
          <h2>Chats</h2>
          <div class="panel-actions">
            <button class="icon-btn" id="documents-btn" type="button">Documents</button>
            <button class="icon-btn" id="collapse-left-btn" type="button" aria-expanded="true">Hide</button>
          </div>
        </div>
        <button class="new-chat-btn" id="new-chat-btn" type="button">New Chat</button>
      </div>
      <div class="session-list" id="session-list"></div>
    </aside>

    <section class="main">
      <div class="status-chip compact-status" id="status-chip"><span class="pulse"></span><span id="status-text">Initializing...</span></div>
      <div class="messages" id="messages"></div>
      <footer class="composer">
        <div class="composer-panel">
          <textarea id="prompt" placeholder="Ask a question, or paste a note to turn into a reply." disabled></textarea>
          <div class="actions">
            <button class="secondary" id="reset-btn" type="button" disabled>Reset</button>
            <button class="primary" id="send-btn" type="button" disabled>Send</button>
          </div>
        </div>
      </footer>
    </section>

    <div class="panel-rail panel-rail-right">
      <button class="rail-btn" id="open-right-btn" type="button" aria-expanded="true">Runtime</button>
    </div>

    <aside class="sidebar">
      <div class="card">
        <div class="card-head">
          <h2>Runtime</h2>
          <button class="icon-btn" id="collapse-right-btn" type="button" aria-expanded="true">Hide</button>
        </div>
        <div class="state-row">
          <div class="badge" id="runtime-badge"><span class="dot"></span><span id="runtime-status">initializing</span></div>
        </div>
        <div class="kv">
          <div><strong>Detail</strong><br><span id="runtime-detail">Loading components.</span></div>
          <div><strong>Session</strong><br><span id="session-id">waiting for session</span></div>
          <div><strong>Debug log</strong><br><span id="log-path">not available yet</span></div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h2>Live Trace</h2>
          <span class="card-subtle">Current turn only</span>
        </div>
        <div class="muted" id="trace-empty">No response yet.</div>
        <div class="trace-list" id="trace-list" hidden></div>
      </div>
    </aside>
  </div>

  <div class="modal-backdrop" id="documents-modal" hidden>
    <div class="modal-shell" role="dialog" aria-modal="true" aria-labelledby="documents-title">
      <div class="modal-header">
        <div>
          <h2 id="documents-title">Documents</h2>
          <div class="muted">Upload files into knowledge/ or delete them from the index. Sync uses the existing incremental document pipeline.</div>
        </div>
        <button class="icon-btn" id="close-documents-btn" type="button">Close</button>
      </div>
      <div class="modal-toolbar">
        <label class="upload-btn">
          <input id="document-file-input" type="file" accept=".txt,.pdf" hidden />
          <span>Upload document</span>
        </label>
        <div class="muted" id="documents-status">No document action yet.</div>
      </div>
      <div class="document-list" id="document-list"></div>
    </div>
  </div>

  <script>
    const shell = document.querySelector('.shell');
    const sessionsPanel = document.querySelector('.sessions');
    const runtimePanel = document.querySelector('.sidebar');
    const messages = document.getElementById('messages');
    const prompt = document.getElementById('prompt');
    const sendBtn = document.getElementById('send-btn');
    const resetBtn = document.getElementById('reset-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const documentsBtn = document.getElementById('documents-btn');
    const collapseLeftBtn = document.getElementById('collapse-left-btn');
    const collapseRightBtn = document.getElementById('collapse-right-btn');
    const openLeftBtn = document.getElementById('open-left-btn');
    const openRightBtn = document.getElementById('open-right-btn');
    const sessionList = document.getElementById('session-list');
    const runtimeStatus = document.getElementById('runtime-status');
    const runtimeDetail = document.getElementById('runtime-detail');
    const runtimeBadge = document.getElementById('runtime-badge');
    const statusText = document.getElementById('status-text');
    const sessionIdLabel = document.getElementById('session-id');
    const logPath = document.getElementById('log-path');
    const traceEmpty = document.getElementById('trace-empty');
    const traceList = document.getElementById('trace-list');
    const documentsModal = document.getElementById('documents-modal');
    const closeDocumentsBtn = document.getElementById('close-documents-btn');
    const documentFileInput = document.getElementById('document-file-input');
    const documentList = document.getElementById('document-list');
    const documentsStatus = document.getElementById('documents-status');

    const initialSessionId = window.__INITIAL_SESSION_ID__ || '';
    const STORAGE_KEYS = {
      leftCollapsed: 'roots:leftCollapsed',
      rightCollapsed: 'roots:rightCollapsed',
    };

    let activeSessionId = initialSessionId;
    let currentMessages = [];
    let sessionsCache = [];
    let documentsCache = [];
    let ready = false;
    let generating = false;
    let autoScrollPinned = true;
    let healthTimer = null;
    let documentBusy = false;
    let leftCollapsed = localStorage.getItem(STORAGE_KEYS.leftCollapsed) === '1';
    let rightCollapsed = localStorage.getItem(STORAGE_KEYS.rightCollapsed) === '1';

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function relativeTimeLabel(isoTimestamp) {
      const timestamp = new Date(isoTimestamp);
      if (Number.isNaN(timestamp.getTime())) {
        return '';
      }
      const deltaMinutes = Math.max(0, Math.round((Date.now() - timestamp.getTime()) / 60000));
      if (deltaMinutes < 1) return 'just now';
      if (deltaMinutes < 60) return `${deltaMinutes}m ago`;
      const hours = Math.round(deltaMinutes / 60);
      if (hours < 24) return `${hours}h ago`;
      return `${Math.round(hours / 24)}d ago`;
    }

    function formatFileSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
      return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }

    function isNearBottom() {
      return messages.scrollHeight - messages.scrollTop - messages.clientHeight < 120;
    }

    function scrollToLatest() {
      messages.scrollTop = messages.scrollHeight;
    }

    function resizePrompt() {
      prompt.style.height = '72px';
      const maxHeight = Math.min(window.innerHeight * 0.42, 420);
      const targetHeight = Math.min(prompt.scrollHeight, maxHeight);
      prompt.style.height = `${targetHeight}px`;
      prompt.style.overflowY = prompt.scrollHeight > maxHeight ? 'auto' : 'hidden';
    }

    function setTraceEmpty(text) {
      traceEmpty.hidden = false;
      traceList.hidden = true;
      traceList.innerHTML = '';
      traceEmpty.textContent = text;
    }

    function setGenerating(nextValue) {
      generating = nextValue;
      sendBtn.disabled = !ready || generating;
      sendBtn.textContent = generating ? 'Generating...' : 'Send';
      sendBtn.classList.toggle('busy', generating);
    }

    function setDocumentBusy(text) {
      documentBusy = Boolean(text);
      documentsStatus.innerHTML = text ? `<span class="busy-pill">${escapeHtml(text)}</span>` : 'No document action yet.';
      documentFileInput.disabled = documentBusy;
      closeDocumentsBtn.disabled = documentBusy;
    }

    function setLayoutState() {
      shell.classList.toggle('left-collapsed', leftCollapsed);
      shell.classList.toggle('right-collapsed', rightCollapsed);
      sessionsPanel.classList.toggle('is-collapsed', leftCollapsed);
      runtimePanel.classList.toggle('is-collapsed', rightCollapsed);
      collapseLeftBtn.textContent = leftCollapsed ? 'Show' : 'Hide';
      collapseRightBtn.textContent = rightCollapsed ? 'Show' : 'Hide';
      openLeftBtn.textContent = leftCollapsed ? 'Chats' : 'Hide Chats';
      openRightBtn.textContent = rightCollapsed ? 'Runtime' : 'Hide Runtime';
      collapseLeftBtn.setAttribute('aria-expanded', String(!leftCollapsed));
      collapseRightBtn.setAttribute('aria-expanded', String(!rightCollapsed));
      openLeftBtn.setAttribute('aria-expanded', String(!leftCollapsed));
      openRightBtn.setAttribute('aria-expanded', String(!rightCollapsed));
      localStorage.setItem(STORAGE_KEYS.leftCollapsed, leftCollapsed ? '1' : '0');
      localStorage.setItem(STORAGE_KEYS.rightCollapsed, rightCollapsed ? '1' : '0');
    }

    function appendMessage(role, content, options = {}) {
      const shouldScroll = options.shouldScroll ?? autoScrollPinned;
      const item = document.createElement('article');
      item.className = `message ${role}`;
      const roleLabel = document.createElement('div');
      roleLabel.className = 'role';
      roleLabel.textContent = role;
      const messageContent = document.createElement('div');
      messageContent.className = 'content';
      if (role === 'assistant' && options.renderedContent != null) {
        messageContent.innerHTML = options.renderedContent;
      } else {
        messageContent.textContent = content;
      }
      item.append(roleLabel, messageContent);
      messages.appendChild(item);
      if (shouldScroll) scrollToLatest();
      return item;
    }

    function renderMessages(messageList) {
      const shouldScroll = autoScrollPinned;
      messages.innerHTML = '';
      if (!messageList.length) {
        appendMessage('system', 'This chat is empty. Start typing to add the first message.', { shouldScroll: false });
        if (shouldScroll) scrollToLatest();
        return;
      }
      for (const message of messageList) {
        appendMessage(message.role, message.content, {
          shouldScroll: false,
          renderedContent: message.rendered_content,
        });
      }
      if (shouldScroll) scrollToLatest();
    }

    function renderTrace(payload) {
      traceList.innerHTML = '';
      const memory = payload?.retrieval_metadata?.memory;
      const documents = payload?.retrieval_metadata?.documents;
      const documentResult = payload?.document_result;

      if (!memory && !documents && !documentResult) {
        setTraceEmpty('No live turn yet. The trace updates after a reply is generated.');
        return;
      }

      traceEmpty.hidden = true;
      traceList.hidden = false;

      const memoryCard = document.createElement('div');
      memoryCard.className = 'trace-item';
      memoryCard.innerHTML = `
        <div class="head"><strong>Facts used</strong><span>${memory?.retrieved ? 'retrieved' : 'not used'}</span></div>
        <div>${memory?.facts?.length ? memory.facts.map((fact) => `<div>${escapeHtml(fact)}</div>`).join('') : '<div class="muted">No memory facts injected.</div>'}</div>
      `;
      traceList.appendChild(memoryCard);

      const docsCard = document.createElement('div');
      docsCard.className = 'trace-item';
      const chunkHtml = documents?.chunks?.length
        ? documents.chunks.map((chunk) => `
            <div class="trace-item" style="margin-top:10px; background: rgba(255,255,255,0.03);">
              <div class="head"><strong>${escapeHtml(chunk.source)}</strong><span>${chunk.injected ? 'used' : 'held back'}</span></div>
              <div class="pill-row">
                ${chunk.location ? `<span class="pill">${escapeHtml(chunk.location)}</span>` : ''}
                <span class="pill">score available in log</span>
              </div>
              <div class="chunk-text">${escapeHtml(chunk.text)}</div>
            </div>
          `).join('')
        : '<div class="muted">No document chunks were used for this turn.</div>';
      docsCard.innerHTML = `
        <div class="head"><strong>Retrieved chunks</strong><span>${documents?.retrieved ? 'retrieved' : 'not used'}</span></div>
        <div class="muted">Current turn only. Scores are shown in the debug log.</div>
        ${chunkHtml}
      `;
      traceList.appendChild(docsCard);

      const resultCard = document.createElement('div');
      resultCard.className = 'trace-item';
      resultCard.innerHTML = `
        <div class="head"><strong>Document result</strong><span>${documentResult?.routed_relevant ? 'routed' : 'not routed'}</span></div>
        <div class="muted">${escapeHtml(documentResult?.reason || 'No reason provided.')}</div>
      `;
      traceList.appendChild(resultCard);
    }

    function renderSessions(sessions) {
      sessionsCache = sessions;
      sessionList.innerHTML = '';

      if (!sessions.length) {
        sessionList.innerHTML = '<div class="panel-empty">No saved chats yet.</div>';
        return;
      }

      for (const session of sessions) {
        const item = document.createElement('div');
        item.className = `session-item${session.id === activeSessionId ? ' active' : ''}`;
        item.dataset.sessionId = session.id;
        item.title = `${session.title} · ${relativeTimeLabel(session.updated_at) || session.updated_at}`;
        item.innerHTML = `
          <div class="session-title">${escapeHtml(session.title)}</div>
          <button class="session-delete" type="button" aria-label="Delete conversation">×</button>
        `;

        item.addEventListener('click', async (event) => {
          if (event.target && event.target.closest('.session-delete')) {
            return;
          }
          await loadSession(session.id);
        });

        item.querySelector('.session-delete').addEventListener('click', async (event) => {
          event.stopPropagation();
          await deleteSession(session.id);
        });

        sessionList.appendChild(item);
      }
    }

    function syncSessionRow(session, moveToTop = false) {
      const existing = sessionList.querySelector(`[data-session-id="${session.id}"]`);
      if (!existing) {
        return;
      }
      existing.classList.toggle('active', session.id === activeSessionId);
      existing.querySelector('.session-title').textContent = session.title;
      existing.title = `${session.title} · ${relativeTimeLabel(session.updated_at) || session.updated_at}`;
      if (moveToTop) {
        sessionList.prepend(existing);
      }
    }

    async function refreshSessions() {
      const response = await fetch('/sessions', { cache: 'no-store' });
      const payload = await response.json();
      renderSessions(payload.sessions || []);
    }

    async function refreshDocuments() {
      const response = await fetch('/documents', { cache: 'no-store' });
      const payload = await response.json();
      documentsCache = payload.documents || [];
      renderDocuments(documentsCache);
    }

    function renderDocuments(documents) {
      documentList.innerHTML = '';
      if (!documents.length) {
        documentList.innerHTML = '<div class="document-empty">No documents are indexed yet. Upload a .txt or .pdf to begin.</div>';
        return;
      }

      for (const item of documents) {
        const row = document.createElement('div');
        row.className = 'document-row';
        row.innerHTML = `
          <div>
            <div class="document-name">${escapeHtml(item.filename)}</div>
            <div class="document-meta">${escapeHtml(item.file_size_label)} · ${escapeHtml(relativeTimeLabel(item.modified_at) || item.modified_at)}</div>
          </div>
          <div class="document-status">${escapeHtml(item.summary)}</div>
          <div class="document-meta">Indexed ${escapeHtml(relativeTimeLabel(item.created_at) || item.created_at)}</div>
          <div class="document-actions">
            <button class="document-delete" type="button">Delete</button>
          </div>
        `;

        row.querySelector('.document-delete').addEventListener('click', async () => {
          await deleteDocument(item.id, item.filename);
        });

        documentList.appendChild(row);
      }
    }

    async function activateSession(sessionIdValue) {
      await fetch(`/sessions/${encodeURIComponent(sessionIdValue)}/activate`, { method: 'POST' });
    }

    async function loadSession(sessionIdValue, syncCookie = true) {
      const response = await fetch(`/sessions/${encodeURIComponent(sessionIdValue)}`, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`Unable to load session ${sessionIdValue}`);
      }
      const payload = await response.json();
      activeSessionId = payload.session.id;
      currentMessages = payload.messages || [];
      sessionIdLabel.textContent = `${payload.session.title} · ${payload.session.id.slice(0, 8)}`;
      renderMessages(currentMessages);
      renderSessions(sessionsCache.map((session) => ({
        ...session,
        active: session.id === activeSessionId,
      })));
      setTraceEmpty('No retrieval metadata is stored for historical chats. Live turns still populate the trace panel.');
      if (syncCookie) {
        await activateSession(activeSessionId);
      }
    }

    async function createNewChat() {
      const response = await fetch('/sessions', { method: 'POST' });
      const payload = await response.json();
      activeSessionId = payload.session.id;
      currentMessages = [];
      sessionIdLabel.textContent = `${payload.session.title} · ${payload.session.id.slice(0, 8)}`;
      renderMessages(currentMessages);
      setTraceEmpty('No response yet.');
      await activateSession(activeSessionId);
      await refreshSessions();
      prompt.focus();
    }

    async function deleteSession(sessionIdValue) {
      if (!confirm('Delete this conversation? This cannot be undone.')) {
        return;
      }

      const response = await fetch(`/sessions/${encodeURIComponent(sessionIdValue)}`, { method: 'DELETE' });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        setTraceEmpty(detail.detail || 'Unable to delete the conversation.');
        return;
      }

      sessionsCache = sessionsCache.filter((session) => session.id !== sessionIdValue);
      if (activeSessionId === sessionIdValue) {
        activeSessionId = '';
        currentMessages = [];
        renderMessages(currentMessages);
        sessionIdLabel.textContent = 'no active session';
        if (sessionsCache.length) {
          await loadSession(sessionsCache[0].id, true);
        } else {
          await createNewChat();
        }
      } else {
        renderSessions(sessionsCache);
      }
    }

    async function uploadDocument(file) {
      if (!file) return;
      setDocumentBusy('Uploading and re-indexing...');
      try {
        const formData = new FormData();
        formData.append('file', file, file.name);
        const response = await fetch('/documents/upload', { method: 'POST', body: formData });
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.detail || 'Upload failed');
        }
        const payload = await response.json();
        documentsCache = payload.documents || [];
        renderDocuments(documentsCache);
        documentsStatus.innerHTML = '<span class="busy-pill">Upload complete and re-indexed.</span>';
      } catch (error) {
        documentsStatus.innerHTML = `<span class="busy-pill">Upload failure: ${escapeHtml(String(error))}</span>`;
      } finally {
        documentFileInput.value = '';
        setDocumentBusy('');
      }
    }

    async function deleteDocument(documentId, filename) {
      if (!confirm(`Delete ${filename}? This removes the file from knowledge/ and re-indexes documents.`)) {
        return;
      }

      setDocumentBusy('Deleting and re-indexing...');
      try {
        const response = await fetch(`/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' });
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.detail || 'Delete failed');
        }
        const payload = await response.json();
        documentsCache = payload.documents || [];
        renderDocuments(documentsCache);
        documentsStatus.innerHTML = '<span class="busy-pill">Delete complete and re-indexed.</span>';
      } catch (error) {
        documentsStatus.innerHTML = `<span class="busy-pill">Delete failure: ${escapeHtml(String(error))}</span>`;
      } finally {
        setDocumentBusy('');
      }
    }

    function openDocumentsModal() {
      documentsModal.hidden = false;
      document.body.style.overflow = 'hidden';
      refreshDocuments().catch((error) => {
        documentsStatus.innerHTML = `<span class="busy-pill">Unable to load documents: ${escapeHtml(String(error))}</span>`;
      });
    }

    function closeDocumentsModal() {
      documentsModal.hidden = true;
      document.body.style.overflow = '';
    }

    async function updateHealth() {
      try {
        const response = await fetch('/health', { cache: 'no-store' });
        const payload = await response.json();
        runtimeStatus.textContent = payload.status;
        runtimeDetail.textContent = payload.detail || 'No detail provided.';
        statusText.textContent = payload.status === 'ready' ? 'Ready' : payload.status === 'failed' ? 'Failed' : 'Initializing...';
        logPath.textContent = payload.log_path || 'not available yet';

        runtimeBadge.style.borderColor = payload.status === 'ready' ? 'rgba(125, 227, 208, 0.28)' : payload.status === 'failed' ? 'rgba(245, 185, 113, 0.28)' : 'rgba(255,255,255,0.08)';
        runtimeBadge.querySelector('.dot').style.background = payload.status === 'ready' ? 'var(--accent)' : payload.status === 'failed' ? 'var(--accent-2)' : 'var(--accent-3)';

        ready = payload.status === 'ready';
        prompt.disabled = !ready;
        setGenerating(generating && ready);
        resetBtn.disabled = !ready;
        if (payload.status === 'failed') {
          appendMessage('system', `Startup failed: ${payload.error || 'unknown error'}`, { shouldScroll: false });
        }
      } catch (error) {
        runtimeStatus.textContent = 'offline';
        runtimeDetail.textContent = String(error);
        statusText.textContent = 'Offline';
        ready = false;
        prompt.disabled = true;
        setGenerating(false);
        resetBtn.disabled = true;
      }
    }

    async function sendMessage() {
      const message = prompt.value.trim();
      if (!message || !ready || generating) {
        return;
      }

      if (!activeSessionId) {
        await createNewChat();
      }

      autoScrollPinned = isNearBottom();
      prompt.value = '';
      resizePrompt();
      const userBubble = appendMessage('user', message, { shouldScroll: autoScrollPinned });
      const pending = appendMessage('assistant', 'Generating...', { shouldScroll: autoScrollPinned });
      pending.querySelector('.role').innerHTML = '<span class="spinner">Generating</span>';
      setGenerating(true);
      prompt.disabled = true;

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, session_id: activeSessionId }),
        });
        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.detail || payload.error || 'Request failed');
        }

        pending.querySelector('.content').innerHTML = payload.rendered_content || '';
        pending.querySelector('.role').textContent = 'assistant';
        activeSessionId = payload.session_id || activeSessionId;
        currentMessages.push(
          { role: 'user', content: message },
          { role: 'assistant', content: payload.reply || '', rendered_content: payload.rendered_content || '' },
        );
        sessionIdLabel.textContent = `${payload.session?.title || 'Chat'} · ${activeSessionId.slice(0, 8)}`;
        syncSessionRow(payload.session || { id: activeSessionId, title: payload.session?.title || 'Chat', updated_at: new Date().toISOString() }, true);
        renderTrace(payload);
        if (autoScrollPinned) {
          scrollToLatest();
        }
      } catch (error) {
        pending.querySelector('.role').textContent = 'assistant';
        pending.querySelector('.content').textContent = `Request failed: ${error}`;
      } finally {
        prompt.disabled = false;
        setGenerating(false);
        prompt.focus();
        if (autoScrollPinned) {
          scrollToLatest();
        }
      }
    }

    async function resetConversation() {
      if (!ready) {
        return;
      }
      await fetch('/reset', { method: 'POST' });
      setTraceEmpty('Turn context reset for the active session. Historical messages remain in chat_sessions.db.');
    }

    sendBtn.addEventListener('click', sendMessage);
    resetBtn.addEventListener('click', resetConversation);
    newChatBtn.addEventListener('click', createNewChat);
    documentsBtn.addEventListener('click', openDocumentsModal);
    closeDocumentsBtn.addEventListener('click', closeDocumentsModal);
    documentsModal.addEventListener('click', (event) => {
      if (event.target === documentsModal) {
        closeDocumentsModal();
      }
    });
    documentFileInput.addEventListener('change', async () => {
      const [file] = documentFileInput.files || [];
      await uploadDocument(file);
    });
    collapseLeftBtn.addEventListener('click', () => {
      leftCollapsed = !leftCollapsed;
      setLayoutState();
    });
    collapseRightBtn.addEventListener('click', () => {
      rightCollapsed = !rightCollapsed;
      setLayoutState();
    });
    openLeftBtn.addEventListener('click', () => {
      leftCollapsed = !leftCollapsed;
      setLayoutState();
    });
    openRightBtn.addEventListener('click', () => {
      rightCollapsed = !rightCollapsed;
      setLayoutState();
    });
    prompt.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });
    prompt.addEventListener('input', resizePrompt);
    messages.addEventListener('scroll', () => {
      autoScrollPinned = isNearBottom();
    });

    async function loadInitialState() {
      setLayoutState();
      setGenerating(false);
      resizePrompt();
      appendMessage('system', 'Chat is ready when the model finishes loading.', { shouldScroll: false });
      try {
        const response = await fetch('/sessions', { cache: 'no-store' });
        const payload = await response.json();
        sessionsCache = payload.sessions || [];
        renderSessions(sessionsCache);
        if (activeSessionId && sessionsCache.some((session) => session.id === activeSessionId)) {
          await loadSession(activeSessionId, false);
        } else if (sessionsCache.length) {
          await loadSession(sessionsCache[0].id, false);
        } else {
          await createNewChat();
        }
      } catch (error) {
        setTraceEmpty(`Unable to load session history: ${error}`);
      }
      await updateHealth();
      if (healthTimer) {
        clearInterval(healthTimer);
      }
      healthTimer = setInterval(updateHealth, 8000);
    }

    loadInitialState();
  </script>
</body>
</html>
"""
  return html.replace("__INITIAL_SESSION_ID_PLACEHOLDER__", initial_session_id)


@app.on_event("startup")
def startup_event() -> None:
    threading.Thread(target=_initialize_app, daemon=True).start()


@app.get("/health")
def health() -> dict:
    return _get_state_snapshot()


@app.get("/sessions")
def list_sessions() -> dict:
    return {"sessions": _list_session_records()}


@app.post("/sessions")
def create_session() -> dict:
    return {"session": _create_session_record(DEFAULT_SESSION_TITLE)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return _session_payload(session_id)


@app.post("/sessions/{session_id}/activate")
def activate_session(session_id: str, response: Response) -> dict:
    session_record = _activate_browser_session(session_id, response=response)
    return {"session": session_record}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
  deleted = _delete_session_record(session_id)
  if not deleted:
    raise HTTPException(status_code=404, detail="Session not found")
  return {"status": "ok", "deleted": session_id}


@app.get("/documents")
def list_documents() -> dict:
  return {"documents": _list_documents()}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
  if not file.filename:
    raise HTTPException(status_code=400, detail="Missing filename")

  try:
    content = await file.read(DOCUMENT_UPLOAD_MAX_BYTES + 1)
    saved_path = _save_uploaded_document(file.filename, content)
    sync_summary = _sync_documents()
    return {
      "status": "ok",
      "saved": saved_path.name,
      "sync": sync_summary,
      "documents": _list_documents(),
    }
  except HTTPException:
    raise
  except Exception as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
  removed_path = _delete_document_file(document_id)
  sync_summary = _sync_documents()
  return {
    "status": "ok",
    "deleted": document_id,
    "removed_file": removed_path.name,
    "sync": sync_summary,
    "documents": _list_documents(),
  }


@app.get("/", response_class=HTMLResponse)
def root(request: Request, response: Response):
    session_id = _get_or_create_session_id(request, response)
    response = HTMLResponse(_format_trace_html(session_id))
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


@app.post("/chat")
async def chat(request: Request, payload: ChatRequest):
    with APP_STATE.init_lock:
        if APP_STATE.status == "failed":
            raise HTTPException(status_code=503, detail=APP_STATE.error or "Application failed to initialize")
        if APP_STATE.status != "ready" or APP_STATE.orchestrator is None:
            raise HTTPException(status_code=503, detail="Application is still initializing")

    response = JSONResponse({"status": "ok"})
    session_id = payload.session_id or request.cookies.get(SESSION_COOKIE)
    if not session_id or not _session_exists(session_id):
      session_record = _create_session_record(_title_from_message(payload.message))
      session_id = session_record["id"]
      response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        path="/",
      )
    session_state = _get_session(session_id)

    result = await asyncio.to_thread(_process_turn, session_id, session_state, payload.message)
    response = JSONResponse(result)
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


@app.post("/reset")
def reset(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id or not _session_exists(session_id):
        return {"status": "ok", "reset": False}

    with APP_STATE.init_lock:
        APP_STATE.sessions[session_id] = SessionState(session_id=session_id)
    return {"status": "ok", "reset": True}


def _process_turn(session_id: str, session_state: SessionState, message: str) -> dict:
    with APP_STATE.generation_lock:
        orchestrator = APP_STATE.orchestrator
        logger = APP_STATE.logger
        if orchestrator is None or logger is None:
            raise HTTPException(status_code=503, detail="Application is not ready")

        session_state.turn_number += 1
        turn_number = session_state.turn_number

        with capture_prints(logger):
            messages, reply, memory_context, document_result, retrieval_metadata = orchestrator.process_turn(
                message,
                session_state.messages,
                turn_number=turn_number,
                maintain_history=True,
                reply_postprocess=strip_speaker_tags,
                include_retrieval_metadata=True,
            )

        session_state.messages = messages

        if session_state.turn_number == 1 and session_state.title == DEFAULT_SESSION_TITLE:
            session_state.title = _title_from_message(message)

        _persist_turn(
            session_id,
            turn_number,
            message,
            reply,
            title_source=session_state.title if session_state.turn_number == 1 else None,
        )

        return {
            "status": "ok",
            "session_id": session_id,
          "session": _get_session_record(session_id),
            "turn_number": turn_number,
            "reply": reply,
            "rendered_content": _render_assistant_markdown(reply),
            "memory_context": memory_context,
            "document_result": _serialize_document_result(document_result),
            "retrieval_metadata": _serialize_retrieval_metadata(retrieval_metadata),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp:app", host="127.0.0.1", port=8000, reload=False)
