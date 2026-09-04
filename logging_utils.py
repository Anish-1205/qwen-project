"""Configure diagnostics, terminal styling, and sensitive tool-log redaction."""

from __future__ import annotations

import contextlib
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import re
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from app_paths import DEBUG_LOG_PATH
from tools.config import MAX_LOG_PAYLOAD_CHARS

try:
    from colorama import just_fix_windows_console
except Exception:  # pragma: no cover - optional dependency
    just_fix_windows_console = None

if just_fix_windows_console is not None:
    just_fix_windows_console()


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"


DEFAULT_LOG_PATH = DEBUG_LOG_PATH


def _wrap(*codes: str) -> str:
    return "".join(codes)


def styled(text: str, *codes: str) -> str:
    return f"{_wrap(*codes)}{text}{RESET}"


def prompt_text(text: str) -> str:
    return styled(text, BOLD, CYAN)


def assistant_label(text: str) -> str:
    return styled(text, BOLD)


def dim_text(text: str) -> str:
    return styled(text, DIM)


def status_text(text: str) -> str:
    return dim_text(text)


def turn_status_text(text: str) -> str:
    return dim_text(text)


def debug_separator() -> str:
    return dim_text("" )


_TAG_PATTERN = re.compile(r"\[([^\]]+)\]")
_EVENT_PATTERN = re.compile(r"^(?:\[Turn (\d+)\]\s*)?\[([^\]]+)\]\s*(.*)$", re.DOTALL)

_SENSITIVE_FIELD = re.compile(r"(?:token|api_?key|access_?key|private_?key|secret|password|passwd|authorization|cookie|credential|session_?id)", re.I)
_URL_IN_TEXT = re.compile(r"https?://[^\s\"'<>]+", re.I)
_QUERY_VALUE = re.compile(r"([?&][^=\s&]+)=([^\s&#]*)")


def _sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[malformed URL redacted]"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "[invalid URL redacted]" if "://" in value or "@" in value or "?" in value else value
    try:
        port = parsed.port
    except ValueError:
        return urlunsplit((parsed.scheme, parsed.hostname, parsed.path, "[redacted]" if parsed.query else "", ""))
    host = parsed.hostname
    if port:
        host = f"{host}:{port}"
    # Redact the complete query: key-only queries are commonly bearer tokens.
    query = "[redacted]" if parsed.query else ""
    return urlunsplit((parsed.scheme, host, parsed.path, query, ""))


def sanitize_tool_log_payload(payload, limit: int = MAX_LOG_PAYLOAD_CHARS) -> str:
    """Redact secrets/URL query values and bound tool log payloads."""
    def clean(value, key: str = ""):
        if _SENSITIVE_FIELD.search(key):
            return "[redacted]"
        if isinstance(value, dict):
            output = {}
            for child_key, child_value in value.items():
                if child_key in {"text", "rows", "entries"} and isinstance(child_value, (str, list)):
                    output[child_key] = f"[{len(child_value)} {'characters' if isinstance(child_value, str) else 'items'}]"
                else:
                    output[child_key] = clean(child_value, str(child_key))
            return output
        if isinstance(value, list):
            return [clean(item, key) for item in value[:50]]
        if isinstance(value, str):
            if "url" in key.casefold():
                return _sanitize_url(value.strip())
            sanitized = _URL_IN_TEXT.sub(lambda match: _sanitize_url(match.group(0)), value)
            return _QUERY_VALUE.sub(r"\1=[redacted]", sanitized)
        return value
    rendered = json.dumps(clean(payload), ensure_ascii=False, default=str, sort_keys=True)
    if len(rendered) <= limit:
        return rendered
    suffix = "…[truncated]"
    if limit <= len(suffix):
        return suffix[: max(0, limit)]
    return rendered[: max(0, limit - len(suffix))] + suffix


class DebugLogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(datefmt="%H:%M:%S")
        self._last_turn = None

    def _color_level(self, levelname: str) -> str:
        if levelname == "WARNING":
            return YELLOW
        if levelname == "ERROR" or levelname == "CRITICAL":
            return RED
        return ""

    def _emphasize_tags(self, text: str) -> str:
        return _TAG_PATTERN.sub(lambda match: styled(match.group(0), BOLD), text)

    def _style_lines(self, message: str) -> str:
        lines = message.splitlines() or [message]
        if len(lines) > 1:
            first = lines[0]
            tail = [f"{DIM}{line}{RESET}" if line.strip() else line for line in lines[1:]]
            return "\n".join([first] + tail)

        if message.startswith("└─") or message.startswith("Source:") or message.startswith("    "):
            return f"{DIM}{message}{RESET}"
        return message

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        event_match = _EVENT_PATTERN.match(message)
        turn_number = int(event_match.group(1)) if event_match and event_match.group(1) else None
        starts_new_turn = False
        if turn_number is not None:
            if self._last_turn is not None and turn_number != self._last_turn:
                starts_new_turn = True
            self._last_turn = turn_number
        prefix = f"{self.formatTime(record, self.datefmt)} | {record.levelname:<7}"
        if event_match:
            category = event_match.group(2)
            detail = event_match.group(3).strip()
            heading = f"TURN {turn_number} · {category}" if turn_number is not None else category
            if detail.startswith("{"):
                try:
                    detail = json.dumps(json.loads(detail), ensure_ascii=False, indent=2, sort_keys=True)
                except ValueError:
                    pass
            detail_lines = detail.splitlines() if detail else []
            rendered = f"{prefix} | {heading}"
            if detail_lines:
                rendered += "\n" + "\n".join(f"    {line}" for line in detail_lines)
        else:
            rendered = f"{prefix} | {message}"
        if record.exc_info:
            rendered += "\n" + self.formatException(record.exc_info)
        return ("\n" if starts_new_turn else "") + rendered


def setup_debug_logger(log_path: str | Path = DEFAULT_LOG_PATH) -> tuple[logging.Logger, Path]:
    resolved_path = Path(log_path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("chatbot_debug")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = RotatingFileHandler(resolved_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(DebugLogFormatter())
    logger.addHandler(handler)
    return logger, resolved_path


class LoggerStream(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)
        return len(text)

    def flush(self) -> None:
        text = self._buffer.strip()
        if text:
            self.logger.log(self.level, text)
        self._buffer = ""


@contextlib.contextmanager
def capture_prints(logger: logging.Logger, level: int = logging.INFO):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stream = LoggerStream(logger, level=level)
    sys.stdout = stream
    sys.stderr = stream
    try:
        yield
    finally:
        stream.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def launch_log_tailer(log_path: str | Path, logger: logging.Logger | None = None) -> bool:
    resolved_path = Path(log_path).resolve()
    if os.name != "nt":
        if logger is not None:
            logger.warning("Log tailing is only auto-launched on Windows; writing logs to %s", resolved_path)
        return False

    commands = [
        ["powershell", "-NoLogo", "-NoProfile", "-Command", f"Get-Content -Path '{resolved_path}' -Wait -Tail 20"],
        ["pwsh", "-NoLogo", "-NoProfile", "-Command", f"Get-Content -Path '{resolved_path}' -Wait -Tail 20"],
    ]
    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    for command in commands:
        try:
            subprocess.Popen(command, creationflags=creation_flags)
            if logger is not None:
                logger.info("Started log tailer for %s", resolved_path)
            return True
        except Exception:
            continue

    if logger is not None:
        logger.warning("Unable to open a live log terminal; logs are available at %s", resolved_path)
    return False
