"""Central, bounded configuration for untrusted tool requests and results."""

from __future__ import annotations

import os
import math


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, minimum), maximum)


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MAX_TOOL_CALLS_PER_TURN = _bounded_int("TOOLS_MAX_CALLS_PER_TURN", 8, 1, 32)
WEB_REQUEST_TIMEOUT = _bounded_float("TOOLS_WEB_REQUEST_TIMEOUT", 12.0, 1.0, 60.0)
WEB_MAX_RESPONSE_BYTES = _bounded_int("TOOLS_WEB_MAX_RESPONSE_BYTES", 2_000_000, 16_384, 20_000_000)
WEB_MAX_REDIRECTS = _bounded_int("TOOLS_WEB_MAX_REDIRECTS", 3, 0, 8)
WEB_MAX_URL_CHARS = _bounded_int("TOOLS_WEB_MAX_URL_CHARS", 4_096, 256, 16_384)
MAX_FILE_BYTES = _bounded_int("TOOLS_MAX_FILE_BYTES", 5_000_000, 1_024, 100_000_000)
MAX_RESULT_CHARS = _bounded_int("TOOLS_MAX_RESULT_CHARS", 20_000, 1_000, 200_000)
MAX_TOOL_CONTEXT_CHARS = _bounded_int("TOOLS_MAX_CONTEXT_CHARS", 60_000, 2_000, 250_000)
CALCULATOR_MAX_EXPRESSION_LENGTH = _bounded_int("TOOLS_CALCULATOR_MAX_EXPRESSION_LENGTH", 200, 20, 2_000)
CALCULATOR_MAX_ABS_EXPONENT = _bounded_int("TOOLS_CALCULATOR_MAX_ABS_EXPONENT", 1_000, 10, 10_000)
CALCULATOR_MAX_VALUES = _bounded_int("TOOLS_CALCULATOR_MAX_VALUES", 100_000, 1, 1_000_000)
CALCULATOR_MAX_RESULT_BITS = _bounded_int("TOOLS_CALCULATOR_MAX_RESULT_BITS", 16_384, 1_024, 131_072)
FILE_CSV_PREVIEW_ROWS = _bounded_int("TOOLS_FILE_CSV_PREVIEW_ROWS", 200, 1, 5_000)
FILE_XLSX_PREVIEW_ROWS = _bounded_int("TOOLS_FILE_XLSX_PREVIEW_ROWS", 50, 1, 1_000)
SPREADSHEET_MAX_RETURNED_ROWS = _bounded_int("TOOLS_SPREADSHEET_MAX_RETURNED_ROWS", 200, 1, 5_000)
MAX_ARCHIVE_UNCOMPRESSED_BYTES = _bounded_int("TOOLS_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 50_000_000, 1_000_000, 500_000_000)
MAX_ARCHIVE_ENTRIES = _bounded_int("TOOLS_MAX_ARCHIVE_ENTRIES", 10_000, 100, 100_000)
DIRECTORY_MAX_ENTRIES = _bounded_int("TOOLS_DIRECTORY_MAX_ENTRIES", 500, 1, 10_000)
DIRECTORY_MAX_DEPTH = _bounded_int("TOOLS_DIRECTORY_MAX_DEPTH", 4, 0, 20)
SPREADSHEET_MAX_FILE_BYTES = _bounded_int("TOOLS_SPREADSHEET_MAX_FILE_BYTES", 20_000_000, 1_024, 200_000_000)
SPREADSHEET_MAX_ROWS = _bounded_int("TOOLS_SPREADSHEET_MAX_ROWS", 100_000, 1, 1_000_000)
SPREADSHEET_MAX_CELLS = _bounded_int("TOOLS_SPREADSHEET_MAX_CELLS", 1_000_000, 1, 10_000_000)
MAX_LOG_PAYLOAD_CHARS = _bounded_int("TOOLS_MAX_LOG_PAYLOAD_CHARS", 2_000, 200, 20_000)
ALLOW_PRIVATE_WEB_HOSTS = _env_bool("TOOLS_ALLOW_PRIVATE_WEB_HOSTS", False)
WEATHER_MAX_FORECAST_DAYS = _bounded_int("TOOLS_WEATHER_MAX_FORECAST_DAYS", 7, 1, 14)
