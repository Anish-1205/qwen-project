"""Bounded structured web search through the Tavily Search API."""

from __future__ import annotations

import html
import json
import os
import re
from urllib.parse import urlsplit

import requests

from . import config
from .common import ToolError


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_FRESHNESS = {"day", "week", "month", "year"}
_HTML_TAG = re.compile(r"<[^>]+>")


def _bounded_text(value, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    text = html.unescape(_HTML_TAG.sub(" ", value))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maximum]


def _read_bounded_json(response: requests.Response) -> dict:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=16_384):
        if not chunk:
            continue
        total += len(chunk)
        if total > config.WEB_SEARCH_MAX_RESPONSE_BYTES:
            raise ToolError("invalid_response", "The search provider returned an oversized response.")
        chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ToolError("invalid_response", "The search provider returned malformed data.") from exc
    if not isinstance(payload, dict):
        raise ToolError("invalid_response", "The search provider returned malformed data.")
    return payload


def _result_url(value) -> str | None:
    if not isinstance(value, str) or len(value) > config.WEB_MAX_URL_CHARS:
        return None
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return value.strip()


def search_web(
    query: str,
    count: int | None = None,
    freshness: str | None = None,
) -> dict:
    """Search Tavily's web index and return bounded result metadata."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ToolError(
            "missing_api_key",
            "Web search is not configured. Set TAVILY_API_KEY to enable it.",
        )
    cleaned_query = re.sub(r"\s+", " ", query).strip() if isinstance(query, str) else ""
    if not cleaned_query or len(cleaned_query) > config.WEB_SEARCH_MAX_QUERY_CHARS:
        raise ToolError("invalid_search_parameter", "query is empty or exceeds the configured length.")
    if count is None:
        count = config.WEB_SEARCH_DEFAULT_RESULTS
    if type(count) is not int or not 1 <= count <= config.WEB_SEARCH_MAX_RESULTS:
        raise ToolError("invalid_search_parameter", "count is outside the allowed range.")
    if freshness is not None and freshness not in _FRESHNESS:
        raise ToolError("invalid_search_parameter", "freshness has an invalid value.")

    request_body = {
        "query": cleaned_query,
        "max_results": count,
        "search_depth": "basic",
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "auto_parameters": False,
    }
    if freshness:
        request_body["time_range"] = freshness

    response = None
    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            json=request_body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "LocalAssistant/1.0",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=config.WEB_REQUEST_TIMEOUT,
            stream=True,
        )
        if response.status_code in {429, 432, 433}:
            raise ToolError("provider_rate_limited", "The search provider rate limit was reached.")
        if response.status_code in {401, 403}:
            raise ToolError("provider_auth_error", "The search provider rejected the configured API key.")
        response.raise_for_status()
        payload = _read_bounded_json(response)
    except requests.Timeout as exc:
        raise ToolError("network_timeout", "The web-search request timed out.") from exc
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise ToolError(
            "provider_error",
            "The search provider request failed.",
            {"status_code": status} if status is not None else {},
        ) from exc
    finally:
        if response is not None:
            response.close()

    provider_results = payload.get("results")
    if not isinstance(provider_results, list):
        raise ToolError("invalid_response", "The search provider returned incomplete data.")

    results = []
    for item in provider_results:
        if len(results) >= count:
            break
        if not isinstance(item, dict):
            continue
        title = _bounded_text(item.get("title"), config.WEB_SEARCH_MAX_TITLE_CHARS)
        url = _result_url(item.get("url"))
        if not title or url is None:
            continue
        normalized = {
            "title": title,
            "url": url,
            "snippet": _bounded_text(item.get("content"), config.WEB_SEARCH_MAX_SNIPPET_CHARS),
        }
        if isinstance(item.get("type"), str):
            normalized["result_type"] = _bounded_text(item["type"], 100)
        if isinstance(item.get("published_date"), str):
            normalized["published"] = _bounded_text(item["published_date"], 100)
        results.append(normalized)

    return {
        "query": cleaned_query,
        "results": results,
        "provider": "Tavily Search API",
        "metadata": {
            "requested_count": count,
            "returned_count": len(results),
            "result_limit": config.WEB_SEARCH_MAX_RESULTS,
            "provider_url": "https://tavily.com/",
        },
    }
