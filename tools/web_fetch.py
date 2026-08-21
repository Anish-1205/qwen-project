from __future__ import annotations

import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from . import config
from .common import ToolError, truncate_text, validate_public_url

_TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml+xml")
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def fetch_webpage(url: str) -> dict:
    current = validate_public_url(url)
    session = requests.Session()
    response = None
    try:
        for redirect_count in range(config.WEB_MAX_REDIRECTS + 1):
            response = session.get(current, timeout=config.WEB_REQUEST_TIMEOUT, stream=True, allow_redirects=False,
                                   headers={"User-Agent": "LocalAssistant/1.0"})
            status_code = getattr(response, "status_code", 0)
            is_redirect = bool(response.is_redirect or response.is_permanent_redirect or status_code in _REDIRECT_STATUSES)
            if is_redirect:
                if redirect_count >= config.WEB_MAX_REDIRECTS:
                    raise ToolError("too_many_redirects", "The URL exceeded the redirect limit.")
                location = response.headers.get("location")
                if not location:
                    raise ToolError("invalid_redirect", "The server returned a redirect without a destination.")
                current = validate_public_url(urljoin(current, location))
                response.close()
                continue
            break
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type and not content_type.startswith(_TEXT_TYPES):
            raise ToolError("unsupported_content_type", "The URL returned unsupported binary content.", {"content_type": content_type})
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=16_384):
            if not chunk:
                continue
            size += len(chunk)
            if size > config.WEB_MAX_RESPONSE_BYTES:
                raise ToolError("response_too_large", "The web response exceeded the configured byte limit.")
            chunks.append(chunk)
        encoding = response.encoding or "utf-8"
        raw = b"".join(chunks).decode(encoding, errors="replace")
    except ToolError:
        raise
    except requests.Timeout as exc:
        raise ToolError("network_timeout", "The web request timed out.") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        raise ToolError("http_error", "The server returned an HTTP error.", {"status": status}) from exc
    except requests.RequestException as exc:
        raise ToolError("network_error", "The web request failed.", {"exception": type(exc).__name__}) from exc
    finally:
        if response is not None:
            response.close()
        session.close()

    title = None
    if "html" in content_type or raw.lstrip().lower().startswith(("<!doctype html", "<html")):
        soup = BeautifulSoup(raw, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else None
        if soup.title:
            soup.title.decompose()
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
    else:
        text = raw
    text = re.sub(r"\s+", " ", text).strip()
    text, truncated = truncate_text(text)
    return {"text": text, "title": title, "final_url": current, "content_type": content_type or "unknown",
            "response_bytes": size, "truncated": truncated}
