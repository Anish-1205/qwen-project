from __future__ import annotations

import ipaddress
import socket
import zipfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from . import config


class ToolError(ValueError):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def resolve_file(path: str, extensions: set[str], max_bytes: int) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise ToolError("path_not_found", "The requested path does not exist.", {"path": str(candidate)})
    if not candidate.is_file():
        raise ToolError("not_a_file", "The requested path is not a regular file.", {"path": str(candidate)})
    if candidate.suffix.lower() not in extensions:
        raise ToolError("unsupported_format", f"Unsupported file extension: {candidate.suffix or '(none)'}")
    size = candidate.stat().st_size
    if size > max_bytes:
        raise ToolError("file_too_large", "The file exceeds the configured size limit.", {"bytes": size, "limit": max_bytes})
    return candidate


def truncate_text(text: str, limit: int = config.MAX_RESULT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def validate_zip_archive(path: Path) -> None:
    """Reject malformed or excessively expanded DOCX/XLSX containers before parsing."""
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > config.MAX_ARCHIVE_ENTRIES:
                raise ToolError("archive_too_large", "The document archive contains too many entries.")
            expanded = sum(entry.file_size for entry in entries)
            if expanded > config.MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ToolError("archive_too_large", "The document exceeds the uncompressed size limit.",
                                {"bytes": expanded, "limit": config.MAX_ARCHIVE_UNCOMPRESSED_BYTES})
    except zipfile.BadZipFile as exc:
        raise ToolError("file_read_error", "The document is not a valid ZIP-based office file.") from exc


def validate_public_url(url: str) -> str:
    if not isinstance(url, str) or not url or len(url) > config.WEB_MAX_URL_CHARS:
        raise ToolError("invalid_url", "The URL is empty or exceeds the configured length limit.")
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise ToolError("invalid_url", "The URL contains invalid control characters.")
    try:
        parsed = urlsplit(url)
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid_url", "The URL is malformed.") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ToolError("invalid_url", "Only absolute http:// and https:// URLs are supported.")
    if parsed.username is not None or parsed.password is not None:
        raise ToolError("invalid_url", "URLs containing embedded credentials are not allowed.")
    try:
        parsed.hostname.encode("idna")
    except UnicodeError as exc:
        raise ToolError("invalid_url", "The URL host is malformed.") from exc
    try:
        port = parsed.port
    except ValueError as exc:
        raise ToolError("invalid_url", "The URL contains an invalid port.") from exc
    if not config.ALLOW_PRIVATE_WEB_HOSTS:
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, port or 0, type=socket.SOCK_STREAM)}
        except (socket.gaierror, UnicodeError) as exc:
            raise ToolError("dns_error", "The URL host could not be resolved.") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global or ip.is_multicast or getattr(ip, "is_site_local", False):
                raise ToolError("blocked_host", "Private, loopback, link-local, reserved, and other non-public hosts are blocked.")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, ""))
