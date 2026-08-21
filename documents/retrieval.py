from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from orchestrator import DocumentRetrievalResult

from .config import DOCUMENT_DB_PATH, SIMILARITY_THRESHOLD, TOP_K
from .db import open_db


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    filepath: str
    file_type: str
    page_number: int | None
    line_start: int | None
    line_end: int | None
    text: str
    score: float


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denominator = (np.linalg.norm(lhs) * np.linalg.norm(rhs)) + 1e-8
    return float(np.dot(lhs, rhs) / denominator)


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""

    rendered: list[str] = []
    for chunk in chunks:
        source_line = f"Source: {chunk.filename}"
        if chunk.page_number is not None:
            source_line += f", page {chunk.page_number}"
        elif chunk.line_start is not None and chunk.line_end is not None:
            source_line += f", lines {chunk.line_start}-{chunk.line_end}"
        rendered.append(f"{source_line}\n{chunk.text}")
    return "\n\n".join(rendered)


def retrieve_relevant_chunks(
    query: str,
    embed_model,
    *,
    db_path: str | Path = DOCUMENT_DB_PATH,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = TOP_K,
    filenames: tuple[str, ...] | list[str] | None = None,
) -> list[RetrievedChunk]:
    with open_db(db_path) as conn:
        cursor = conn.cursor()
        sql = """
            SELECT c.id, c.document_id, d.filename, d.filepath, d.file_type,
                   c.page_number, c.line_start, c.line_end, c.text, c.embedding
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'indexed'
        """
        params: list[str] = []
        if filenames:
            normalized_filenames = [str(filename).casefold() for filename in filenames]
            sql += f" AND LOWER(d.filename) IN ({','.join(['?'] * len(normalized_filenames))})"
            params.extend(normalized_filenames)
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    if not rows:
        return []

    query_vector = np.asarray(embed_model.encode(query), dtype=np.float32)
    scored: list[RetrievedChunk] = []
    for chunk_id, document_id, filename, filepath, file_type, page_number, line_start, line_end, text, embedding_blob in rows:
        if not embedding_blob:
            continue
        chunk_vector = np.frombuffer(embedding_blob, dtype=np.float32)
        if chunk_vector.size == 0:
            continue
        score = _cosine_similarity(query_vector, chunk_vector)
        if score < similarity_threshold:
            continue
        scored.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                filename=filename,
                filepath=filepath,
                file_type=file_type,
                page_number=page_number,
                line_start=line_start,
                line_end=line_end,
                text=text,
                score=score,
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def build_document_result(chunks: list[RetrievedChunk], *, routed_relevant: bool = False, reason: str = "") -> DocumentRetrievalResult:
    if not chunks:
        return DocumentRetrievalResult(context="", routed_relevant=routed_relevant, retrieved_count=0, reason=reason)

    context = format_retrieved_chunks(chunks)
    sources = []
    for chunk in chunks:
        if chunk.page_number is not None:
            source = f"{chunk.filename} (page {chunk.page_number})"
        elif chunk.line_start is not None and chunk.line_end is not None:
            source = f"{chunk.filename} (lines {chunk.line_start}-{chunk.line_end})"
        else:
            source = chunk.filename
        if source not in sources:
            sources.append(source)
    return DocumentRetrievalResult(
        context=context,
        routed_relevant=routed_relevant,
        retrieved_count=len(chunks),
        sources=sources,
        reason=reason,
    )
