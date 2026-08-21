from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DOCUMENT_DB_PATH, ROUTER_KEYWORDS, ROUTER_KEYWORD_BOOST, ROUTER_SIMILARITY_THRESHOLD
from .db import open_db


@dataclass(slots=True)
class RoutingDecision:
    relevant: bool
    score: float
    keyword_boost: float
    reason: str
    matched_filenames: tuple[str, ...] = ()


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denominator = (np.linalg.norm(lhs) * np.linalg.norm(rhs)) + 1e-8
    return float(np.dot(lhs, rhs) / denominator)


def route_document_query(
    query: str,
    embed_model,
    *,
    db_path: str | Path = DOCUMENT_DB_PATH,
    threshold: float = ROUTER_SIMILARITY_THRESHOLD,
    keyword_boost: float = ROUTER_KEYWORD_BOOST,
) -> RoutingDecision:
    query_lower = query.lower()

    with open_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.embedding, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'indexed'
            """
        )
        rows = cursor.fetchall()

    if not rows:
        return RoutingDecision(False, 0.0, 0.0, "no indexed documents")

    indexed_filenames = {str(filename).casefold() for _, filename in rows if filename}
    requested_filenames = {
        match.group(0).casefold()
        for match in re.finditer(r"\b[\w][\w.-]*\.(?:pdf|txt)\b", query, flags=re.I)
    }
    matched_requested = indexed_filenames.intersection(requested_filenames)
    matched_requested.update(filename for filename in indexed_filenames if filename in query_lower)
    if requested_filenames and not matched_requested:
        return RoutingDecision(False, 0.0, 0.0, "named document not indexed")

    query_vector = np.asarray(embed_model.encode(query), dtype=np.float32)
    best_score = 0.0
    matched_filenames: set[str] = set(matched_requested)
    for embedding_blob, filename in rows:
        if not embedding_blob:
            continue
        chunk_vector = np.frombuffer(embedding_blob, dtype=np.float32)
        if chunk_vector.size == 0:
            continue
        score = _cosine_similarity(query_vector, chunk_vector)
        if score > best_score:
            best_score = score
        filename_lower = filename.lower() if filename else ""
        stem_lower = Path(filename_lower).stem if filename_lower else ""
        if filename_lower and (filename_lower in query_lower or stem_lower in query_lower):
            matched_filenames.add(filename_lower)

    boost = 0.0
    if any(keyword in query_lower for keyword in ROUTER_KEYWORDS):
        boost += keyword_boost
    if matched_filenames:
        boost += keyword_boost * 1.5

    adjusted = best_score + boost
    if adjusted >= threshold:
        reason = "filename mention" if matched_filenames else "semantic similarity"
        return RoutingDecision(True, adjusted, boost, reason, tuple(sorted(matched_filenames)))

    return RoutingDecision(False, adjusted, boost, "below routing threshold", tuple(sorted(matched_filenames)))
