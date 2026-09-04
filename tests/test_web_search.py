from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import requests

from intent_classifier import DeterministicIntentRouter
from harness import (
    ConversationOrchestrator,
    DEFAULT_SYSTEM_PROMPT,
    SEARCH_CONFIGURATION_REQUIRED_RESPONSE,
)
from tools import ToolCall, ToolDefinition, ToolManager, ToolRegistry
from tools.common import ToolError
from tools.web_search import TAVILY_SEARCH_URL, search_web
from tests.test_intent_and_orchestration import FakeMemory, StubBackend


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.closed = False
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def iter_content(self, chunk_size=16384):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def close(self):
        self.closed = True


class QueueOrchestrator(ConversationOrchestrator):
    def __init__(self, generated, *, tool_manager=None, logger=None):
        self.generated = list(generated)
        self.inputs = []
        self.overrides = []
        super().__init__(
            StubBackend(), FakeMemory(), tool_manager=tool_manager,
            logger=logger or (lambda message: None),
        )

    def generate_reply(self, messages, **overrides):
        self.inputs.append([dict(message) for message in messages])
        self.overrides.append(dict(overrides))
        if not self.generated:
            raise AssertionError("unexpected model generation")
        return self.generated.pop(0)


def _search_payload(count=3):
    results = []
    for index in range(count):
        results.append({
            "title": f"Result <b>{index}</b>",
            "url": f"https://example{index}.com/page",
            "content": f"A <strong>bounded</strong> snippet for result {index}.",
            "type": "search_result",
            "published_date": f"2026-08-{20 - index:02d}T10:00:00Z",
        })
    return {"results": results, "response_time": 1.2}


def test_search_web_success_is_structured_bounded_and_secret_free():
    response = FakeResponse(_search_payload(4))
    secret = "test-secret-must-not-leak"

    with patch.dict("os.environ", {"TAVILY_API_KEY": secret}, clear=False), patch(
        "tools.web_search.requests.post", return_value=response
    ) as get:
        result = search_web("  current   Python release  ", count=2, freshness="week")

    assert result["query"] == "current Python release"
    assert result["provider"] == "Tavily Search API"
    assert result["metadata"]["requested_count"] == 2
    assert result["metadata"]["returned_count"] == 2
    assert len(result["results"]) == 2
    assert result["results"][0] == {
        "title": "Result 0",
        "url": "https://example0.com/page",
        "snippet": "A bounded snippet for result 0.",
        "result_type": "search_result",
        "published": "2026-08-20T10:00:00Z",
    }
    assert response.closed is True
    assert get.call_args.args == (TAVILY_SEARCH_URL,)
    assert get.call_args.kwargs["json"] == {
        "query": "current Python release", "max_results": 2,
        "search_depth": "basic", "topic": "general", "time_range": "week",
        "include_answer": False, "include_raw_content": False,
        "include_images": False, "auto_parameters": False,
    }
    assert get.call_args.kwargs["headers"]["Authorization"] == f"Bearer {secret}"
    assert secret not in json.dumps(result)
    schema_text = json.dumps(ToolManager().schemas())
    assert "TAVILY_API_KEY" not in schema_text
    search_schema = next(
        schema["function"]["parameters"] for schema in ToolManager().schemas()
        if schema["function"]["name"] == "search_web"
    )
    assert set(search_schema["properties"]) == {"query", "count", "freshness"}


def test_search_web_missing_key_is_structured_and_does_not_call_network():
    with patch.dict("os.environ", {}, clear=True), patch("tools.web_search.requests.post") as get:
        result = ToolManager().execute(ToolCall("search_web", {"query": "latest Python release"}))
    assert not result.ok
    assert result.error_details["code"] == "missing_api_key"
    assert "TAVILY_API_KEY" in result.error
    get.assert_not_called()


@pytest.mark.parametrize(
    ("response", "side_effect", "expected_code"),
    [
        (FakeResponse({}, status_code=429), None, "provider_rate_limited"),
        (FakeResponse({}, status_code=503), None, "provider_error"),
        (FakeResponse({}, status_code=401), None, "provider_auth_error"),
        (None, requests.Timeout(), "network_timeout"),
        (FakeResponse(b"not-json"), None, "invalid_response"),
        (FakeResponse({"unexpected": []}), None, "invalid_response"),
    ],
)
def test_search_web_normalizes_provider_failures(response, side_effect, expected_code):
    with patch.dict("os.environ", {"TAVILY_API_KEY": "configured"}, clear=False), patch(
        "tools.web_search.requests.post", return_value=response, side_effect=side_effect
    ):
        result = ToolManager().execute(ToolCall("search_web", {"query": "test query"}))
    assert not result.ok and result.error_details["code"] == expected_code


def test_search_web_rejects_oversized_response_and_bounds_result_count():
    oversized = FakeResponse(b"{" + b"x" * 100 + b"}")
    with patch.dict("os.environ", {"TAVILY_API_KEY": "configured"}, clear=False), patch(
        "tools.web_search.config.WEB_SEARCH_MAX_RESPONSE_BYTES", 32
    ), patch("tools.web_search.requests.post", return_value=oversized):
        error = ToolManager().execute(ToolCall("search_web", {"query": "test query"}))
    assert not error.ok and error.error_details["code"] == "invalid_response"

    response = FakeResponse(_search_payload(8))
    with patch.dict("os.environ", {"TAVILY_API_KEY": "configured"}, clear=False), patch(
        "tools.web_search.requests.post", return_value=response
    ):
        bounded = ToolManager().execute(ToolCall("search_web", {"query": "test query", "count": 3}))
    assert bounded.ok and len(bounded.data["results"]) == 3
    too_many = ToolManager().execute(ToolCall("search_web", {
        "query": "test query", "count": 1_000,
    }))
    assert not too_many.ok and too_many.error_details["code"] == "validation_error"


def test_search_web_routing_is_actionable_but_not_conceptual():
    router = DeterministicIntentRouter()
    actionable = (
        "Search the web for Python 3.14 changes.",
        "Look up the latest Python release.",
        "Find information online about local language models.",
        "Run a web search for accessible Python tutorials.",
    )
    conceptual = (
        "What is a web search?",
        "How do search engines work?",
        "How can I search the web effectively?",
        "Why is online search useful?",
        "Do not search the web for this.",
    )
    for request in actionable:
        decision = router.analyze(request, [])
        assert decision.tool_use is True and decision.source_for("tool_use") == "deterministic"
    for request in conceptual:
        decision = router.analyze(request, [])
        assert decision.tool_use is False and decision.source_for("tool_use") == "deterministic"


def test_search_web_execution_supplies_structured_results_for_grounding():
    call = '<tool_call>{"name":"search_web","arguments":{"query":"Python 3.14 changes","count":2}}</tool_call>'
    orchestrator = QueueOrchestrator([call, "The returned sources describe Python 3.14 changes."])

    with patch.dict("os.environ", {"TAVILY_API_KEY": "configured"}, clear=False), patch(
        "tools.web_search.requests.post", return_value=FakeResponse(_search_payload(2))
    ):
        reply = orchestrator.generate_tool_aware_reply(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "Search the web for Python 3.14 changes."},
            ],
            turn_number=1,
        )

    assert reply == "The returned sources describe Python 3.14 changes."
    assert orchestrator.last_tool_execution.ok
    assert orchestrator.last_tool_execution.call.name == "search_web"
    tool_message = next(message for message in orchestrator.inputs[-1] if message.get("role") == "tool")
    payload = json.loads(tool_message["content"])
    assert payload["data"]["provider"] == "Tavily Search API"
    assert len(payload["data"]["results"]) == 2


def test_explicit_search_bypasses_model_tool_selection():
    logs = []
    orchestrator = QueueOrchestrator(
        ["The returned sources describe Python 3.14 changes."],
        logger=logs.append,
    )

    with patch.dict("os.environ", {"TAVILY_API_KEY": "configured"}, clear=False), patch(
        "tools.web_search.requests.post", return_value=FakeResponse(_search_payload(2))
    ):
        reply = orchestrator.generate_tool_aware_reply(
            [{"role": "user", "content": "Search the web for Python 3.14 changes."}],
            turn_number=7,
        )

    assert reply == "The returned sources describe Python 3.14 changes."
    assert orchestrator.last_tool_execution.ok
    assert orchestrator.last_tool_execution.call.name == "search_web"
    assert orchestrator.last_tool_execution.call.arguments["query"] == (
        "Search the web for Python 3.14 changes."
    )
    # The only model generation is final synthesis after the deterministic call.
    assert len(orchestrator.inputs) == 1
    assert any(
        "source=deterministic tool=search_web reason=explicit_web_search_request" in message
        for message in logs
    )


def test_missing_key_returns_explicit_configuration_failure_without_model_completion():
    call = '<tool_call>{"name":"search_web","arguments":{"query":"latest Python release"}}</tool_call>'
    orchestrator = QueueOrchestrator([call, "Invented search results."])

    with patch.dict("os.environ", {}, clear=True):
        reply = orchestrator.generate_tool_aware_reply(
            [{"role": "user", "content": "Look up the latest Python release."}],
            turn_number=2,
        )

    assert reply == SEARCH_CONFIGURATION_REQUIRED_RESPONSE
    assert orchestrator.inputs == []
    assert orchestrator.last_tool_execution.error_details["code"] == "missing_api_key"


def test_provider_failure_cannot_be_replaced_by_fabricated_model_results():
    call = '<tool_call>{"name":"search_web","arguments":{"query":"latest Python release"}}</tool_call>'
    orchestrator = QueueOrchestrator([call, "Here are invented results.", "Still invented."])

    with patch.dict("os.environ", {"TAVILY_API_KEY": "configured"}, clear=False), patch(
        "tools.web_search.requests.post", return_value=FakeResponse({}, status_code=503)
    ):
        reply = orchestrator.generate_tool_aware_reply(
            [{"role": "user", "content": "Look up the latest Python release."}],
            turn_number=3,
        )

    assert "no valid registered tool execution" in reply
    assert "invented" not in reply.lower()
    assert len(orchestrator.inputs) == 3


def test_fetch_webpage_cannot_satisfy_a_required_search_action():
    registry = ToolRegistry()
    object_schema = lambda properties, required: {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False,
    }
    registry.register(ToolDefinition(
        "search_web", "Search.", lambda query: {"results": []},
        object_schema({"query": {"type": "string"}}, ["query"]),
    ))
    fetch = Mock(return_value={"text": "page text"})
    registry.register(ToolDefinition(
        "fetch_webpage", "Fetch.", fetch,
        object_schema({"url": {"type": "string"}}, ["url"]),
    ))
    orchestrator = QueueOrchestrator(
        [
            '<tool_call>{"name":"fetch_webpage","arguments":{"url":"https://example.com"}}</tool_call>',
            "I found a result by scraping.",
            "Still claiming success.",
        ],
        tool_manager=ToolManager(registry),
    )

    reply = orchestrator.generate_tool_aware_reply(
        [{"role": "user", "content": "Search the web for Python release information."}],
        turn_number=4,
    )

    assert "no valid registered tool execution" in reply
    assert [result.call.name for result in orchestrator.last_tool_executions] == [
        "search_web", "fetch_webpage",
    ]
    assert orchestrator.last_tool_executions[0].ok
    assert orchestrator.last_tool_executions[1].error_details["code"] == "url_not_in_current_turn"
    assert "fetch_webpage" not in {
        schema["function"]["name"] for schema in orchestrator.overrides[0]["tools"]
    }
    fetch.assert_not_called()


def test_model_selected_search_failure_cannot_fall_back_to_fetch_webpage():
    registry = ToolRegistry()
    schema = lambda name: {
        "type": "object", "properties": {name: {"type": "string"}},
        "required": [name], "additionalProperties": False,
    }

    def failed_search(query):
        raise ToolError("provider_error", "Search failed.")

    registry.register(ToolDefinition("search_web", "Search.", failed_search, schema("query")))
    fetch = Mock(return_value={"text": "page text"})
    registry.register(ToolDefinition(
        "fetch_webpage", "Fetch.", fetch, schema("url"),
    ))
    orchestrator = QueueOrchestrator(
        [
            '<tool_call>{"name":"search_web","arguments":{"query":"recent topic"}}</tool_call>',
            '<tool_call>{"name":"fetch_webpage","arguments":{"url":"https://example.com"}}</tool_call>',
            "I replaced the failed search with scraping.",
            "Still claiming success.",
        ],
        tool_manager=ToolManager(registry),
    )

    reply = orchestrator.generate_tool_aware_reply(
        [{"role": "user", "content": "Handle this external lookup."}],
        turn_number=5,
    )

    assert "no valid registered tool execution" in reply
    assert [result.call.name for result in orchestrator.last_tool_executions] == ["search_web", "fetch_webpage"]
    assert not orchestrator.last_tool_executions[0].ok
    assert not orchestrator.last_tool_executions[1].ok
    assert orchestrator.last_tool_executions[1].error_details["code"] == "url_not_in_current_turn"
    fetch.assert_not_called()


def test_fetch_webpage_executes_only_current_turn_urls_and_exposes_schema():
    registry = ToolRegistry()
    schema = {
        "type": "object", "properties": {"url": {"type": "string"}},
        "required": ["url"], "additionalProperties": False,
    }
    fetch = Mock(side_effect=lambda url: {"url": url, "text": "page text"})
    registry.register(ToolDefinition("fetch_webpage", "Fetch.", fetch, schema))
    orchestrator = QueueOrchestrator(
        [
            '<tool_call>{"name":"fetch_webpage","arguments":{"url":"https://EXAMPLE.com/one"}}</tool_call>'
            '<tool_call>{"name":"fetch_webpage","arguments":{"url":"https://example.org/two"}}</tool_call>',
            "The two supplied pages were read.",
        ],
        tool_manager=ToolManager(registry),
    )

    with patch("tools.manager.validate_public_url", side_effect=lambda value: value):
        reply = orchestrator.generate_tool_aware_reply(
            [{"role": "user", "content": "Compare https://example.com/one and https://example.org/two."}],
            turn_number=6,
        )

    assert reply == "The two supplied pages were read."
    assert [result.ok for result in orchestrator.last_tool_executions] == [True, True]
    assert fetch.call_count == 2
    assert "fetch_webpage" in {
        item["function"]["name"] for item in orchestrator.overrides[0]["tools"]
    }


def test_fetch_webpage_rejects_invented_and_prior_turn_urls_before_execution():
    registry = ToolRegistry()
    schema = {
        "type": "object", "properties": {"url": {"type": "string"}},
        "required": ["url"], "additionalProperties": False,
    }
    fetch = Mock(return_value={"text": "page text"})
    registry.register(ToolDefinition("fetch_webpage", "Fetch.", fetch, schema))

    invented = QueueOrchestrator(
        [
            '<tool_call>{"name":"fetch_webpage","arguments":{"url":"https://invented.example/"}}</tool_call>',
            "I read it.",
            "I still read it.",
        ],
        tool_manager=ToolManager(registry),
    )
    invented_reply = invented.generate_tool_aware_reply(
        [{"role": "user", "content": "Read https://example.com/allowed."}], turn_number=7,
    )
    assert "no valid registered tool execution" in invented_reply
    assert invented.last_tool_execution.error_details["code"] == "url_not_in_current_turn"

    prior = QueueOrchestrator(
        [
            '<tool_call>{"name":"fetch_webpage","arguments":{"url":"https://example.com/prior"}}</tool_call>',
            "I read the prior link.",
            "I still read it.",
        ],
        tool_manager=ToolManager(registry),
    )
    prior_reply = prior.generate_tool_aware_reply(
        [
            {"role": "user", "content": "https://example.com/prior"},
            {"role": "assistant", "content": "Thanks."},
            {"role": "user", "content": "Read that link."},
        ],
        turn_number=8,
    )
    assert "no valid registered tool execution" in prior_reply
    assert prior.last_tool_execution.error_details["code"] == "url_not_in_current_turn"
    assert "fetch_webpage" not in {
        item["function"]["name"] for item in prior.overrides[0]["tools"]
    }
    fetch.assert_not_called()
