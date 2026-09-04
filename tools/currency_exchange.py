"""Bounded daily reference-rate lookup through the Frankfurter v2 API."""

from __future__ import annotations

import json
import re
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

import requests

from . import config
from .common import ToolError


FRANKFURTER_RATE_URL = "https://api.frankfurter.dev/v2/rate/{base}/{quote}"
_CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")


def _currency_code(value: str, field: str) -> str:
    code = value.strip().upper() if isinstance(value, str) else ""
    if not _CURRENCY_CODE.fullmatch(code):
        raise ToolError("invalid_currency", f"{field} must be a three-letter currency code.", {"field": field})
    return code


def _requested_date(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    try:
        parsed = calendar_date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid_date", "date must use YYYY-MM-DD format.") from exc
    if parsed > calendar_date.today():
        raise ToolError("invalid_date", "date must not be in the future.")
    return parsed.isoformat()


def _amount(value: int | float | None) -> Decimal | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ToolError("invalid_amount", "amount must be a finite non-negative number.")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ToolError("invalid_amount", "amount must be a finite non-negative number.") from exc
    maximum = Decimal(str(config.CURRENCY_MAX_AMOUNT))
    if not amount.is_finite() or amount < 0 or amount > maximum:
        raise ToolError(
            "invalid_amount",
            f"amount must be between 0 and {maximum}."
        )
    return amount


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _read_bounded_json(response: requests.Response) -> dict:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=8_192):
        if not chunk:
            continue
        total += len(chunk)
        if total > config.CURRENCY_MAX_RESPONSE_BYTES:
            raise ToolError("invalid_response", "The exchange-rate service returned an oversized response.")
        chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"), parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ToolError("invalid_response", "The exchange-rate service returned malformed data.") from exc
    if not isinstance(payload, dict):
        raise ToolError("invalid_response", "The exchange-rate service returned malformed data.")
    return payload


def currency_exchange(
    base_currency: str,
    quote_currency: str,
    amount: int | float | None = None,
    date: str | None = None,
) -> dict:
    """Return one current or historical daily reference rate and optional conversion."""
    base = _currency_code(base_currency, "base_currency")
    quote = _currency_code(quote_currency, "quote_currency")
    if base == quote:
        raise ToolError("invalid_currency_pair", "base_currency and quote_currency must be different.")
    requested_date = _requested_date(date)
    requested_amount = _amount(amount)
    endpoint = FRANKFURTER_RATE_URL.format(base=base, quote=quote)
    params = {"date": requested_date} if requested_date else {}
    source_url = f"{endpoint}?{urlencode(params)}" if params else endpoint

    response = None
    try:
        response = requests.get(
            endpoint,
            params=params or None,
            headers={"Accept": "application/json", "User-Agent": "LocalAssistant/1.0"},
            timeout=config.WEB_REQUEST_TIMEOUT,
            stream=True,
        )
        if response.status_code == 429:
            raise ToolError("provider_rate_limited", "The exchange-rate service rate limit was reached.")
        response.raise_for_status()
        payload = _read_bounded_json(response)
    except requests.Timeout as exc:
        raise ToolError("network_timeout", "The exchange-rate request timed out.") from exc
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise ToolError(
            "provider_error",
            "The exchange-rate service request failed.",
            {"status_code": status} if status is not None else {},
        ) from exc
    finally:
        if response is not None:
            response.close()

    returned_base = payload.get("base")
    returned_quote = payload.get("quote")
    returned_date = payload.get("date")
    rate = payload.get("rate")
    try:
        parsed_rate = rate if isinstance(rate, Decimal) else Decimal(str(rate))
        calendar_date.fromisoformat(returned_date)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ToolError("invalid_response", "The exchange-rate service returned incomplete data.") from exc
    if returned_base != base or returned_quote != quote or not parsed_rate.is_finite() or parsed_rate <= 0:
        raise ToolError("invalid_response", "The exchange-rate service returned inconsistent data.")

    result = {
        "base_currency": base,
        "quote_currency": quote,
        "rate": _decimal_text(parsed_rate),
        "rate_date": returned_date,
        "provider": "Frankfurter v2",
        "data_type": "daily_reference_rate",
        "source_url": source_url,
    }
    if requested_amount is not None:
        result["amount"] = _decimal_text(requested_amount)
        result["converted_amount"] = _decimal_text(requested_amount * parsed_rate)
    return result
