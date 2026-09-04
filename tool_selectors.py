"""Small, explicit tool-selection role used by the harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import os
from typing import Callable, Sequence


class ToolSelector(ABC):
    """Choose the next allowlisted tool call, or return no calls."""

    identity: str

    @abstractmethod
    def select(
        self,
        messages: Sequence[dict],
        schemas: Sequence[dict],
        tool_results: Sequence[dict] | None = None,
    ) -> str:
        """Return tagged tool calls understood by ``ToolManager``."""


class MainModelToolSelector(ToolSelector):
    identity = "main_model_fallback"

    def __init__(self, generate: Callable[..., str]):
        self._generate = generate

    def select(self, messages, schemas, tool_results=None) -> str:
        return self._generate(messages, tools=list(schemas))


class CompatibleToolSelector(ToolSelector):
    """Compatibility wrapper for existing injected selectors."""

    def __init__(self, selector):
        self._selector = selector
        self.identity = str(getattr(selector, "identity", "injected_selector"))

    def select(self, messages, schemas, tool_results=None) -> str:
        try:
            return self._selector.select(messages, schemas=list(schemas), tool_results=tool_results)
        except TypeError as exc:
            if "tool_results" not in str(exc):
                raise
            return self._selector.select(messages, schemas=list(schemas))


class Needle2ToolSelector(ToolSelector):
    """Native Cactus Needle2 selector; never synthesizes user-facing prose."""

    identity = "needle2"

    def __init__(self, *, minimum_confidence: float = 0.0):
        try:
            import needle
        except ImportError as exc:
            raise RuntimeError(
                "Needle2 is not installed; install requirements-needle.txt or use main_model fallback"
            ) from exc
        self._needle = needle
        self._needle.Needle(tools=[])
        self.minimum_confidence = minimum_confidence
        self._agent = None
        self._schema_fingerprint = ""

    @staticmethod
    def _needle_schemas(schemas: Sequence[dict]) -> list[dict]:
        converted = []
        for schema in schemas:
            function = dict(schema["function"])
            parameters = function.get("parameters", {})
            branches = parameters.get("oneOf") if isinstance(parameters, dict) else None
            if branches:
                properties = {}
                required_sets = []
                variants = []
                for branch in branches:
                    properties.update(branch.get("properties", {}))
                    required = set(branch.get("required", []))
                    required_sets.append(required)
                    if required:
                        variants.append(" + ".join(sorted(required)))
                function["parameters"] = {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(set.intersection(*required_sets)) if required_sets else [],
                    "additionalProperties": False,
                }
                if variants:
                    function["description"] = (
                        f"{function.get('description', '').rstrip()} Valid argument variants: "
                        + " OR ".join(variants)
                        + "."
                    )
            converted.append(function)
        return converted

    def _agent_for(self, schemas: Sequence[dict]):
        direct_schemas = self._needle_schemas(schemas)
        rendered = json.dumps(direct_schemas, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if self._agent is None or fingerprint != self._schema_fingerprint:
            self._agent = self._needle.Needle(tools=direct_schemas)
            self._schema_fingerprint = fingerprint
        return self._agent

    @staticmethod
    def _current_user_text(messages: Sequence[dict]) -> str:
        return next(
            (str(message.get("content", "")) for message in reversed(messages) if message.get("role") == "user"),
            "",
        )

    @staticmethod
    def _tagged_calls(response: dict) -> str:
        if response.get("type") != "call":
            return ""
        validation = response.get("validation")
        if isinstance(validation, dict) and validation.get("ungrounded"):
            return ""
        rendered = []
        for call in response.get("function_calls") or []:
            rendered.append(
                "<tool_call>"
                + json.dumps(
                    {"name": call.get("name"), "arguments": call.get("arguments") or {}},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "</tool_call>"
            )
        return "".join(rendered)

    def select(self, messages, schemas, tool_results=None) -> str:
        agent = self._agent_for(schemas)
        if tool_results is None:
            agent.reset()
            response = agent.complete(self._current_user_text(messages))
        else:
            result_payload = tool_results[0] if len(tool_results) == 1 else list(tool_results)
            response = agent.complete(json.dumps(result_payload, ensure_ascii=True, separators=(",", ":")))
        confidence = response.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < self.minimum_confidence:
            raise RuntimeError(
                f"Needle2 confidence {confidence:.4f} is below threshold {self.minimum_confidence:.4f}"
            )
        if response.get("success") is False:
            raise RuntimeError(f"Needle2 selection failed: {response.get('error_code') or response.get('error')}")
        return self._tagged_calls(response)


def configured_tool_selector(*, logger: Callable[[str], None] = print) -> ToolSelector | None:
    selection = os.environ.get("CHATBOT_TOOL_SELECTOR", "main_model").strip().lower()
    if selection in {"", "main_model", "fallback"}:
        return None
    if selection != "needle2":
        raise ValueError("CHATBOT_TOOL_SELECTOR must be 'main_model' or 'needle2'")
    try:
        threshold = float(os.environ.get("CHATBOT_NEEDLE_MIN_CONFIDENCE", "0"))
        if not 0 <= threshold <= 1:
            raise ValueError
    except ValueError as exc:
        raise ValueError("CHATBOT_NEEDLE_MIN_CONFIDENCE must be between 0 and 1") from exc
    try:
        return Needle2ToolSelector(minimum_confidence=threshold)
    except Exception as exc:
        logger(f"[Tool Selector] Needle2 unavailable; using main-model fallback ({exc}).")
        return None
