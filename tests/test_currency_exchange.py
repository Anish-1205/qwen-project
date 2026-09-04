from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import patch

import pytest
import requests

from intent_classifier import DeterministicIntentRouter
from harness import ConversationOrchestrator, DEFAULT_SYSTEM_PROMPT
from tools import ToolCall, ToolManager
from tools.currency_exchange import FRANKFURTER_RATE_URL, currency_exchange
from tests.test_intent_and_orchestration import FakeMemory, StubBackend


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.closed = False
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def iter_content(self, chunk_size=8192):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def close(self):
        self.closed = True


class QueueOrchestrator(ConversationOrchestrator):
    def __init__(self, generated):
        self.generated = list(generated)
        self.inputs = []
        super().__init__(StubBackend(), FakeMemory(), logger=lambda message: None)

    def generate_reply(self, messages, **overrides):
        self.inputs.append([dict(message) for message in messages])
        if not self.generated:
            raise AssertionError("unexpected model generation")
        return self.generated.pop(0)


def _rate_payload(rate=0.8, *, base="USD", quote="EUR", rate_date="2024-01-02"):
    return {"date": rate_date, "base": base, "quote": quote, "rate": rate}


def test_currency_exchange_registered_schema_and_decimal_conversion():
    response = FakeResponse(_rate_payload(rate=0.8))
    manager = ToolManager()

    with patch("tools.currency_exchange.requests.get", return_value=response) as get:
        result = manager.execute(ToolCall("currency_exchange", {
            "base_currency": "usd",
            "quote_currency": "eur",
            "amount": 12.34,
            "date": "2024-01-02",
        }))

    assert result.ok
    assert result.data == {
        "base_currency": "USD",
        "quote_currency": "EUR",
        "rate": "0.8",
        "rate_date": "2024-01-02",
        "provider": "Frankfurter v2",
        "data_type": "daily_reference_rate",
        "source_url": "https://api.frankfurter.dev/v2/rate/USD/EUR?date=2024-01-02",
        "amount": "12.34",
        "converted_amount": "9.872",
    }
    assert response.closed is True
    assert get.call_args.args == (FRANKFURTER_RATE_URL.format(base="USD", quote="EUR"),)
    assert get.call_args.kwargs["params"] == {"date": "2024-01-02"}
    assert get.call_args.kwargs["stream"] is True


def test_currency_exchange_rate_only_uses_latest_provider_date():
    response = FakeResponse(_rate_payload(rate=83.125, base="USD", quote="INR", rate_date="2026-08-20"))

    with patch("tools.currency_exchange.requests.get", return_value=response) as get:
        result = currency_exchange("USD", "INR")

    assert result["rate"] == "83.125"
    assert result["rate_date"] == "2026-08-20"
    assert "amount" not in result and "converted_amount" not in result
    assert get.call_args.kwargs["params"] is None


def test_currency_exchange_accepts_empty_optional_date_as_latest():
    response = FakeResponse(_rate_payload(rate=0.8))
    manager = ToolManager()

    with patch("tools.currency_exchange.requests.get", return_value=response) as get:
        result = manager.execute(ToolCall("currency_exchange", {
            "base_currency": "USD",
            "quote_currency": "EUR",
            "amount": 100,
            "date": "",
        }))

    assert result.ok
    assert result.validated_arguments["date"] == ""
    assert get.call_args.kwargs["params"] is None


@pytest.mark.parametrize(
    ("arguments", "expected_code"),
    [
        ({"base_currency": "US", "quote_currency": "EUR"}, "validation_error"),
        ({"base_currency": "USD", "quote_currency": "USD"}, "invalid_currency_pair"),
        ({"base_currency": "USD", "quote_currency": "EUR", "date": "not-a-date"}, "invalid_date"),
        ({"base_currency": "USD", "quote_currency": "EUR", "amount": 1_000_000_000_000_001}, "validation_error"),
    ],
)
def test_currency_exchange_rejects_invalid_arguments_before_network(arguments, expected_code):
    with patch("tools.currency_exchange.requests.get") as get:
        result = ToolManager().execute(ToolCall("currency_exchange", arguments))
    assert not result.ok and result.error_details["code"] == expected_code
    get.assert_not_called()


def test_currency_exchange_rejects_future_date_before_network():
    future = (date.today() + timedelta(days=1)).isoformat()
    with patch("tools.currency_exchange.requests.get") as get:
        result = ToolManager().execute(ToolCall("currency_exchange", {
            "base_currency": "USD", "quote_currency": "EUR", "date": future,
        }))
    assert not result.ok and result.error_details["code"] == "invalid_date"
    get.assert_not_called()


@pytest.mark.parametrize(
    ("response", "side_effect", "expected_code"),
    [
        (FakeResponse({}, status_code=429), None, "provider_rate_limited"),
        (FakeResponse({}, status_code=503), None, "provider_error"),
        (None, requests.Timeout(), "network_timeout"),
        (FakeResponse(b"not json"), None, "invalid_response"),
        (FakeResponse(_rate_payload(base="GBP")), None, "invalid_response"),
    ],
)
def test_currency_exchange_returns_structured_provider_failures(response, side_effect, expected_code):
    with patch("tools.currency_exchange.requests.get", return_value=response, side_effect=side_effect):
        result = ToolManager().execute(ToolCall("currency_exchange", {
            "base_currency": "USD", "quote_currency": "EUR",
        }))
    assert not result.ok and result.error_details["code"] == expected_code


def test_currency_exchange_bounds_provider_response():
    response = FakeResponse(b"{" + b"x" * 100 + b"}")
    with patch("tools.currency_exchange.config.CURRENCY_MAX_RESPONSE_BYTES", 32), patch(
        "tools.currency_exchange.requests.get", return_value=response
    ):
        result = ToolManager().execute(ToolCall("currency_exchange", {
            "base_currency": "USD", "quote_currency": "EUR",
        }))
    assert not result.ok and result.error_details["code"] == "invalid_response"


def test_currency_exchange_routing_is_actionable_but_not_conceptual():
    router = DeterministicIntentRouter()
    actionable = (
        "Convert 100 USD to INR.",
        "What is the latest exchange rate for EUR to GBP?",
        "Exchange 50 Canadian dollars into Japanese yen.",
        "How much is 25 USD in EUR?",
    )
    conceptual = (
        "What is an exchange rate?",
        "How do currency exchange rates work?",
        "Why do exchange rates fluctuate?",
        "Explain currency conversion.",
    )
    for request in actionable:
        decision = router.analyze(request, [])
        assert decision.tool_use is True and decision.source_for("tool_use") == "deterministic"
    for request in conceptual:
        decision = router.analyze(request, [])
        assert decision.tool_use is False and decision.source_for("tool_use") == "deterministic"


def test_currency_exchange_result_is_supplied_for_final_grounding():
    call = (
        '<tool_call>{"name":"currency_exchange","arguments":'
        '{"base_currency":"USD","quote_currency":"EUR","amount":100}}</tool_call>'
    )
    orchestrator = QueueOrchestrator([call, "100 USD is 80 EUR at the 2024-01-02 reference rate."])

    with patch("tools.currency_exchange.requests.get", return_value=FakeResponse(_rate_payload(rate=0.8))):
        reply = orchestrator.generate_tool_aware_reply(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "Use the currency_exchange tool to convert 100 USD to EUR."},
            ],
            turn_number=1,
        )

    assert reply == "100 USD is 80 EUR at the 2024-01-02 reference rate."
    assert orchestrator.last_tool_execution.ok
    assert orchestrator.last_tool_execution.call.name == "currency_exchange"
    tool_message = next(message for message in orchestrator.inputs[-1] if message.get("role") == "tool")
    tool_payload = json.loads(tool_message["content"])
    assert tool_payload["data"]["converted_amount"] == "80"
    assert tool_payload["data"]["rate_date"] == "2024-01-02"
    ledger = orchestrator.inputs[-1][-1]["content"]
    assert '"result_id": "result_1"' in ledger
    assert '"provenance": "dedicated"' in ledger
    assert '"converted_amount": "80"' in ledger
