"""Run confidence-routing and weather-tool prompts against the local Qwen model."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intent_classifier import ConfidenceTier
from harness import ConversationOrchestrator, DEFAULT_SYSTEM_PROMPT, OrchestrationOutcome
from models import create_backend, get_model_spec
from tools import ToolDefinition, ToolManager, ToolRegistry


class MemoryStub:
    last_retrieval_stats = {"facts": []}


def weather_manager() -> ToolManager:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "weather",
            "Return fixture weather for a named place.",
            lambda place: {"location": place, "current": {"temperature_c": 24, "condition": "Clear"}},
            {
                "type": "object",
                "properties": {"place": {"type": "string", "minLength": 1}},
                "required": ["place"],
                "additionalProperties": False,
            },
        )
    )
    return ToolManager(registry)


def main() -> int:
    configured = get_model_spec("qwen")
    options = dict(configured.load_options)
    options["model_kwargs"] = {"local_files_only": True}
    backend = create_backend(replace(configured, load_options=options))
    backend.load()

    cases = [
        ("I moved to Bangalore recently, and can you also check the weather there?", [], True, True, "Bangalore"),
        ("I live in Pune now. What's the weather there?", [], True, True, "Pune"),
        ("I'm visiting Tokyo tomorrow; check the weather there.", [], False, True, "Tokyo"),
        ("I moved to Bangalore recently.", [], True, False, None),
        ("Check the weather in Bangalore.", [], False, True, "Bangalore"),
        ("hows the weather in bengaluru ?", [], False, True, "Bengaluru"),
        ("what is the weather in kolkata", [], False, True, "Kolkata"),
        (
            "Can you check the weather there?",
            [{"role": "user", "content": "I live in Pune now."}, {"role": "assistant", "content": "Understood."}],
            False,
            True,
            "Pune",
        ),
        ("Can you check the weather there?", [], False, True, None),
        ("How do weather forecasts work?", [], False, False, None),
    ]
    results = []
    passed = True
    for prompt, history, expected_write, expected_tool, expected_place in cases:
        manager = weather_manager()
        orchestrator = ConversationOrchestrator(
            backend,
            MemoryStub(),
            tool_manager=manager,
            reply_generation_kwargs={"max_new_tokens": 160, "do_sample": False},
            logger=lambda _message: None,
        )
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *history]
        intent = orchestrator.classify_intent(prompt, messages)
        route = orchestrator.last_routing_decision
        reply = ""
        if route.outcome is OrchestrationOutcome.SELECT_TOOL:
            reply = orchestrator.generate_tool_aware_reply(
                [*messages, {"role": "user", "content": prompt}],
                turn_number=1,
            )
        executed_places = [
            result.validated_arguments.get("place")
            for result in orchestrator.last_tool_executions
            if result.ok and result.call.name == "weather" and result.validated_arguments
        ]
        write_evidence = route.evidence_for("memory_write")
        case_pass = bool(
            intent.memory_write is expected_write
            and intent.tool_use is expected_tool
            and (not expected_write or write_evidence.confidence is ConfidenceTier.HIGH)
            and (
                (expected_place is not None and executed_places == [expected_place])
                or (expected_place is None and not executed_places)
            )
            and (
                prompt != "Can you check the weather there?"
                or history
                or route.outcome is OrchestrationOutcome.ASK_USER
            )
        )
        passed = passed and case_pass
        results.append(
            {
                "prompt": prompt,
                "pass": case_pass,
                "intent": {
                    "memory_read": intent.memory_read,
                    "memory_write": intent.memory_write,
                    "tool_use": intent.tool_use,
                    "general_chat": intent.general_chat,
                },
                "outcome": route.outcome.value,
                "resolved_references": orchestrator.last_resolved_tool_references,
                "executed_weather_places": executed_places,
                "reply": reply,
            }
        )
    print(json.dumps({"pass": passed, "cases": results}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
