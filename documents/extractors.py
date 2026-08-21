from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from .config import MIN_PDF_CHARS_PER_PAGE

try:
    from charset_normalizer import from_bytes as detect_encoding
except Exception:  # pragma: no cover - optional dependency fallback
    detect_encoding = None


@dataclass(slots=True)
class ExtractionSegment:
    text: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass(slots=True)
class ExtractionResult:
    text: str
    segments: list[ExtractionSegment] = field(default_factory=list)
    page_count: int | None = None
    char_count: int = 0


Extractor = Callable[[Path], ExtractionResult]
EXTRACTORS: dict[str, Extractor] = {}


def register_extractor(extension: str):
    def decorator(function: Extractor) -> Extractor:
        EXTRACTORS[extension.lower()] = function
        return function

    return decorator


def _normalize_paragraph_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _paragraph_segments(text: str, *, page_number: int | None = None) -> list[ExtractionSegment]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    segments: list[ExtractionSegment] = []
    buffer: list[str] = []
    start_line: int | None = None
    last_line: int | None = None

    def flush() -> None:
        nonlocal buffer, start_line, last_line
        if not buffer:
            return
        paragraph = _normalize_paragraph_text(" ".join(buffer))
        if paragraph:
            segments.append(
                ExtractionSegment(
                    text=paragraph,
                    page_number=page_number,
                    line_start=start_line,
                    line_end=last_line,
                )
            )
        buffer = []
        start_line = None
        last_line = None

    for line_number, raw_line in enumerate(normalized.split("\n"), start=1):
        stripped = raw_line.strip()
        if not stripped:
            flush()
            continue
        if start_line is None:
            start_line = line_number
        buffer.append(stripped)
        last_line = line_number

    flush()
    return segments


def _decode_text(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    if detect_encoding is not None:
        matches = detect_encoding(raw_bytes)
        best = matches.best()
        if best is not None:
            try:
                return str(best)
            except Exception:
                pass

    return raw_bytes.decode("latin-1", errors="ignore")


@register_extractor(".txt")
def extract_txt(path: Path) -> ExtractionResult:
    raw_bytes = path.read_bytes()
    decoded = _decode_text(raw_bytes)
    segments = _paragraph_segments(decoded)
    normalized_text = "\n\n".join(segment.text for segment in segments)
    return ExtractionResult(
        text=normalized_text,
        segments=segments,
        page_count=None,
        char_count=len(normalized_text),
    )


@register_extractor(".pdf")
def extract_pdf(path: Path) -> ExtractionResult:
    reader = PdfReader(str(path))
    segments: list[ExtractionSegment] = []
    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        page_texts.append(extracted)
        segments.extend(_paragraph_segments(extracted, page_number=page_number))

    normalized_text = "\n\n".join(segment.text for segment in segments)
    if len(normalized_text.strip()) < max(MIN_PDF_CHARS_PER_PAGE, 1) * max(len(reader.pages), 1):
        # OCR can be inserted here later for scanned PDFs that do not produce text.
        pass

    return ExtractionResult(
        text=normalized_text,
        segments=segments,
        page_count=len(reader.pages),
        char_count=len(normalized_text),
    )


def get_extractor(file_type: str) -> Extractor | None:
    return EXTRACTORS.get(file_type.lower())
