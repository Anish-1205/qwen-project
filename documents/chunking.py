from __future__ import annotations

import re
from dataclasses import dataclass

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .extractors import ExtractionSegment


@dataclass(slots=True)
class ChunkRecord:
    chunk_index: int
    text: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _token_estimate(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _trim_overlap(sentences: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []

    kept: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        kept.append(sentence)
        total += _token_estimate(sentence)
        if total >= overlap_tokens:
            break
    kept.reverse()
    return kept


def _build_chunk(chunk_sentences: list[tuple[str, ExtractionSegment]], chunk_index: int) -> ChunkRecord:
    texts = [sentence for sentence, _ in chunk_sentences]
    metadata = [segment for _, segment in chunk_sentences]
    page_numbers = {segment.page_number for segment in metadata if segment.page_number is not None}
    line_starts = [segment.line_start for segment in metadata if segment.line_start is not None]
    line_ends = [segment.line_end for segment in metadata if segment.line_end is not None]

    return ChunkRecord(
        chunk_index=chunk_index,
        text=" ".join(texts).strip(),
        page_number=next(iter(page_numbers)) if len(page_numbers) == 1 else (next(iter(page_numbers)) if page_numbers else None),
        line_start=min(line_starts) if line_starts else None,
        line_end=max(line_ends) if line_ends else None,
    )


def chunk_segments(segments: list[ExtractionSegment], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[ChunkRecord]:
    if not segments:
        return []

    chunks: list[ChunkRecord] = []
    current: list[tuple[str, ExtractionSegment]] = []
    current_tokens = 0
    chunk_index = 0
    current_page = segments[0].page_number

    def flush() -> None:
        nonlocal current, current_tokens, chunk_index
        if not current:
            return
        chunk = _build_chunk(current, chunk_index)
        if chunk.text:
            chunks.append(chunk)
            chunk_index += 1

        overlap_sentences = _trim_overlap([sentence for sentence, _ in current], chunk_overlap)
        if overlap_sentences:
            overlap_start = len(current) - len(overlap_sentences)
            current = current[overlap_start:]
            current_tokens = sum(_token_estimate(sentence) for sentence, _ in current)
        else:
            current = []
            current_tokens = 0

    for segment in segments:
        if current and current_page is not None and segment.page_number is not None and segment.page_number != current_page:
            flush()
            current_page = segment.page_number

        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT.split(segment.text) if sentence.strip()]
        if not sentences:
            sentences = [segment.text.strip()]

        for sentence in sentences:
            sentence_tokens = _token_estimate(sentence)
            if current and current_tokens + sentence_tokens > chunk_size:
                flush()
            current.append((sentence, segment))
            current_tokens += sentence_tokens
            current_page = segment.page_number if segment.page_number is not None else current_page

    flush()
    return chunks
