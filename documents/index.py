from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from harness import DocumentRetrievalResult

from .chunking import chunk_segments
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    DOCUMENT_DB_PATH,
    MIN_PDF_CHARS_PER_PAGE,
    ROUTER_SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from .db import init_db, open_db
from .discovery import DiscoveryPlan, classify_documents, document_id_for_path, scan_documents
from .extractors import ExtractionResult, get_extractor
from .retrieval import build_document_result, retrieve_relevant_chunks
from .router import route_document_query


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@dataclass(slots=True)
class DocumentIndex:
    embed_model: Any
    logger: logging.Logger | None = None
    db_path: str | Path = DOCUMENT_DB_PATH
    docs_dir: str | Path = DOCS_DIR
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K
    similarity_threshold: float = SIMILARITY_THRESHOLD
    router_threshold: float = ROUTER_SIMILARITY_THRESHOLD
    min_pdf_chars_per_page: int = MIN_PDF_CHARS_PER_PAGE

    def __post_init__(self) -> None:
        self.docs_dir = Path(self.docs_dir)
        init_db(self.db_path)
        if self.logger is None:
            self.logger = logging.getLogger("chatbot_debug")

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _load_existing_documents(self) -> dict[str, dict]:
        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, filepath, file_type, file_size, content_hash, created_at, modified_at, status FROM documents"
            )
            rows = cursor.fetchall()

        existing: dict[str, dict] = {}
        for row in rows:
            doc_id, filename, filepath, file_type, file_size, content_hash, created_at, modified_at, status = row
            filepath_key = Path(filepath).resolve().as_posix().lower()
            existing[filepath_key] = {
                "id": doc_id,
                "filename": filename,
                "filepath": filepath,
                "filepath_key": filepath_key,
                "file_type": file_type,
                "file_size": file_size,
                "content_hash": content_hash,
                "created_at": created_at,
                "modified_at": modified_at,
                "status": status,
            }
        return existing

    def sync(self) -> DiscoveryPlan:
        try:
            if not self.docs_dir.exists():
                self._log(f"[Documents] Warning: documents directory not found at {self.docs_dir}; continuing without document context.")
                existing = self._load_existing_documents()
                if existing:
                    with open_db(self.db_path) as conn:
                        cursor = conn.cursor()
                        for record in existing.values():
                            cursor.execute("UPDATE documents SET status = 'deleted', modified_at = ? WHERE id = ?", (_utc_now(), record["id"]))
                            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (record["id"],))
                return DiscoveryPlan(missing_directory=True)

            discovered = scan_documents(self.docs_dir)
            existing = self._load_existing_documents()
            plan = classify_documents(discovered, existing)

            for record in plan.deleted:
                self._mark_deleted(record["id"])

            for item in plan.new + plan.changed:
                self._index_file(item)

            self._log(
                f"[Documents] Sync complete: {len(plan.new)} new, {len(plan.changed)} changed, {len(plan.unchanged)} unchanged, {len(plan.deleted)} deleted."
            )
            return plan
        except Exception as exc:
            self._log(f"[Documents] Warning: sync failed but the chatbot will continue: {exc}")
            return DiscoveryPlan(missing_directory=not self.docs_dir.exists())

    def _mark_deleted(self, document_id: str) -> None:
        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE documents SET status = 'deleted', modified_at = ? WHERE id = ?", (_utc_now(), document_id))
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def _upsert_document(self, record, status: str) -> None:
        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents (id, filename, filepath, file_type, file_size, content_hash, created_at, modified_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    filepath = excluded.filepath,
                    file_type = excluded.file_type,
                    file_size = excluded.file_size,
                    content_hash = excluded.content_hash,
                    modified_at = excluded.modified_at,
                    status = excluded.status
                """,
                (
                    record.document_id,
                    record.filename,
                    str(record.filepath),
                    record.file_type,
                    record.file_size,
                    record.content_hash,
                    _utc_now(),
                    _utc_now(),
                    status,
                ),
            )

    def _replace_chunks(self, document_id: str, chunk_rows: list[tuple]) -> None:
        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cursor.executemany(
                "INSERT INTO chunks (id, document_id, chunk_index, page_number, line_start, line_end, text, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                chunk_rows,
            )

    def _index_file(self, item) -> None:
        extractor = get_extractor(item.file_type)
        if extractor is None:
            self._log(f"[Documents] Warning: no extractor registered for {item.filepath.name}; marking as failed.")
            self._upsert_document(item, "failed")
            return

        try:
            extracted: ExtractionResult = extractor(item.filepath)
        except Exception as exc:
            self._log(f"[Documents] Warning: failed to extract {item.filepath.name}: {exc}")
            self._upsert_document(item, "failed")
            return

        if item.file_type == ".pdf":
            page_count = extracted.page_count or 0
            if not extracted.text.strip() or len(extracted.text.strip()) < max(self.min_pdf_chars_per_page, 1) * max(page_count, 1):
                self._log(
                    f"[Documents] Warning: PDF extraction for {item.filepath.name} returned no usable text; marking as empty_extraction. OCR can be added here later."
                )
                self._upsert_document(item, "empty_extraction")
                return

        if not extracted.text.strip():
            self._log(f"[Documents] Warning: {item.filepath.name} produced empty text; marking as empty_extraction.")
            self._upsert_document(item, "empty_extraction")
            return

        try:
            chunks = chunk_segments(extracted.segments, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            if not chunks:
                self._log(f"[Documents] Warning: {item.filepath.name} yielded no chunks; marking as empty_extraction.")
                self._upsert_document(item, "empty_extraction")
                return

            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = np.asarray(self.embed_model.encode(chunk_texts), dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            self._upsert_document(item, "indexed")
            chunk_rows = []
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                chunk_id = hashlib.sha256(f"{item.document_id}:{chunk.chunk_index}:{chunk.text}".encode("utf-8")).hexdigest()
                chunk_rows.append(
                    (
                        chunk_id,
                        item.document_id,
                        chunk.chunk_index,
                        chunk.page_number,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.text,
                        embedding.astype(np.float32).tobytes(),
                        _utc_now(),
                    )
                )

            self._replace_chunks(item.document_id, chunk_rows)
            self._log(f"[Documents] Indexed {item.filepath.name}: {len(chunk_rows)} chunk(s).")
        except Exception as exc:
            self._log(f"[Documents] Warning: failed to index {item.filepath.name}: {exc}")
            self._upsert_document(item, "failed")

    def lookup_context(self, query: str) -> DocumentRetrievalResult:
        with open_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'indexed'")
            indexed_count = cursor.fetchone()[0]

        if indexed_count == 0:
            self._log("[Documents] No indexed documents available; skipping document retrieval.")
            return DocumentRetrievalResult(context="", routed_relevant=False, retrieved_count=0, reason="no indexed documents")

        routing = route_document_query(query, self.embed_model, db_path=self.db_path, threshold=self.router_threshold)
        if not routing.relevant:
            self._log(f"[Documents] Query not routed to documents: {routing.reason}.")
            return DocumentRetrievalResult(context="", routed_relevant=False, retrieved_count=0, reason=routing.reason)

        chunks = retrieve_relevant_chunks(
            query,
            self.embed_model,
            db_path=self.db_path,
            similarity_threshold=self.similarity_threshold,
            top_k=self.top_k,
            filenames=routing.matched_filenames or None,
        )
        if not chunks:
            self._log("[Documents] Query routed to documents, but no chunks cleared the retrieval threshold.")
            return DocumentRetrievalResult(context="", routed_relevant=True, retrieved_count=0, reason="no chunks above similarity threshold")

        chunk_count = len(chunks)
        noun = "chunk" if chunk_count == 1 else "chunks"
        self._log(f"[Document Retrieval] {chunk_count} relevant {noun} found")
        for index, chunk in enumerate(chunks, start=1):
            if chunk.page_number is not None:
                location = f"{chunk.filename} | page {chunk.page_number}"
            elif chunk.line_start is not None and chunk.line_end is not None:
                location = f"{chunk.filename} | lines {chunk.line_start}-{chunk.line_end}"
            else:
                location = chunk.filename
            self._log(f"└─ {index:02d}_{location} | score={chunk.score:.2f}")

        return build_document_result(chunks, routed_relevant=True, reason=routing.reason)
