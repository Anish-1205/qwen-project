from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from harness import DEFAULT_SYSTEM_PROMPT, HarnessRunner, RunRequest
from models import GenerationResult, ModelBackend, ModelCapabilities, ModelSpec
from tool_selectors import Needle2ToolSelector, ToolSelector
from tools import ToolManager
from tools.registry import ToolDefinition, ToolRegistry


class QueueMainModel(ModelBackend):
    def __init__(self, model_name: str, outputs=()):
        self._spec = ModelSpec(
            model_name, model_name, model_name, "test",
            ModelCapabilities(tool_schemas=True, tool_messages=True),
        )
        self.outputs = list(outputs)
        self.calls = []

    @property
    def spec(self):
        return self._spec

    def load(self):
        pass

    def generate(self, request):
        self.calls.append(request)
        return GenerationResult(self.outputs.pop(0))

    def count_tokens(self, messages):
        return 1


class QueueSelector(ToolSelector):
    identity = "needle2"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def select(self, messages, schemas, tool_results=None):
        self.calls.append({
            "messages": list(messages),
            "schemas": list(schemas),
            "tool_results": tool_results,
        })
        return self.outputs.pop(0)


class FakeMemory:
    last_retrieval_stats = {"facts": []}


def calculator_manager(function) -> ToolManager:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        "calculator",
        "Evaluate an arithmetic expression.",
        function,
        {
            "type": "object",
            "properties": {"expression": {"type": "string", "minLength": 1}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    ))
    return ToolManager(registry)


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2.5-3B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"])
def test_dedicated_needle_selects_and_main_model_synthesizes(model_name):
    selector = QueueSelector([
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
        "",
    ])
    main_model = QueueMainModel(model_name, ["The result is 4."])
    runner = HarnessRunner(
        main_model,
        FakeMemory(),
        tool_selector=selector,
        tool_manager=calculator_manager(lambda expression: {"value": 4}),
        logger=lambda message: None,
    )

    result = runner.run(RunRequest(
        "Calculate 2 + 2 using calculator.",
        [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
        turn_number=1,
    ))

    assert result.output == "The result is 4."
    assert len(selector.calls) == 2
    assert selector.calls[1]["tool_results"][0]["data"]["value"] == 4
    assert len(main_model.calls) == 1
    assert main_model.calls[0].tools is None
    selection_events = [event for event in result.trace if event.kind == "tool_selection"]
    assert selection_events[0].data["selector"] == "needle2"
    assert selection_events[0].data["main_model"] == model_name


@pytest.mark.parametrize("model_name", ["qwen", "smollm2"])
def test_main_model_fallback_uses_same_bounded_tool_flow(model_name):
    main_model = QueueMainModel(model_name, [
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
        "The result is 4.",
    ])
    runner = HarnessRunner(
        main_model,
        FakeMemory(),
        tool_manager=calculator_manager(lambda expression: {"value": 4}),
        logger=lambda message: None,
    )

    reply = runner.generate_tool_aware_reply(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Calculate 2 + 2 using calculator."},
        ],
        turn_number=1,
    )

    assert reply == "The result is 4."
    assert main_model.calls[0].tools
    assert main_model.calls[0].do_sample is False
    assert runner.tool_selector_identity == "main_model_fallback"


def test_main_model_selection_logs_active_model_and_protocol_without_raw_output():
    logs = []
    main_model = QueueMainModel("smollm2", [
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
        "The result is 4.",
    ])
    runner = HarnessRunner(
        main_model,
        FakeMemory(),
        tool_manager=calculator_manager(lambda expression: {"value": 4}),
        logger=logs.append,
    )

    runner.generate_tool_aware_reply(
        [{"role": "user", "content": "Calculate 2 + 2 using calculator."}],
        turn_number=3,
    )

    assert any(
        "selector=main_model_fallback model_id=smollm2" in message
        and "parsed_calls=1" in message
        and "tagged_protocol=True" in message
        for message in logs
    )
    selector_logs = [message for message in logs if "[Tool Selector]" in message]
    assert not any("2 + 2" in message for message in selector_logs)


def test_repeated_failed_call_executes_once_then_duplicate_rejections_terminate():
    executions = []

    def fail(expression):
        executions.append(expression)
        raise ValueError("simulated failure")

    repeated = '<tool_call>{"name":"calculator","arguments":{"expression":"1 / 0"}}</tool_call>'
    selector = QueueSelector([repeated, repeated, repeated])
    runner = HarnessRunner(
        QueueMainModel("qwen"),
        FakeMemory(),
        tool_selector=selector,
        tool_manager=calculator_manager(fail),
        logger=lambda message: None,
    )

    result = runner.run(RunRequest(
        "Calculate 1 / 0 using calculator.",
        [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
        turn_number=1,
    ))

    assert executions == ["1 / 0"]
    assert [entry.payload["error"]["code"] for entry in result.state.tool_ledger] == [
        "execution_error", "duplicate_failed_call", "duplicate_failed_call",
    ]
    assert result.output.startswith("I stopped because the same failed tool call")
    assert any(event.kind == "duplicate_rejection" for event in result.trace)
    assert any(
        event.kind == "tool_termination" and event.data["reason"] == "repeated_duplicate_failed_call"
        for event in result.trace
    )


def test_repeated_schema_invalid_call_is_validated_once_then_rejected_as_duplicate():
    repeated = '<tool_call>{"name":"calculator","arguments":{}}</tool_call>'
    selector = QueueSelector([repeated, repeated, repeated])
    manager = calculator_manager(lambda expression: {"value": 0})
    validations = []
    original_validate = manager.validate_call

    def counted_validate(call):
        validations.append(call)
        return original_validate(call)

    manager.validate_call = counted_validate
    runner = HarnessRunner(
        QueueMainModel("qwen"), FakeMemory(), tool_selector=selector,
        tool_manager=manager, logger=lambda message: None,
    )

    result = runner.run(RunRequest(
        "Use calculator.", [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}], turn_number=1,
    ))

    assert len(validations) == 1
    assert [entry.payload["error"]["code"] for entry in result.state.tool_ledger] == [
        "validation_error", "duplicate_failed_call", "duplicate_failed_call",
    ]


def test_changed_arguments_after_failure_are_executed():
    executions = []

    def calculate(expression):
        executions.append(expression)
        if expression == "1 / 0":
            raise ValueError("division by zero")
        return {"value": 2}

    selector = QueueSelector([
        '<tool_call>{"name":"calculator","arguments":{"expression":"1 / 0"}}</tool_call>',
        '<tool_call>{"name":"calculator","arguments":{"expression":"1 + 1"}}</tool_call>',
        "",
    ])
    main_model = QueueMainModel("smollm2", ["The corrected result is 2."])
    runner = HarnessRunner(
        main_model,
        FakeMemory(),
        tool_selector=selector,
        tool_manager=calculator_manager(calculate),
        logger=lambda message: None,
    )

    reply = runner.generate_tool_aware_reply(
        [{"role": "user", "content": "Use calculator and correct any invalid expression."}],
        turn_number=1,
    )

    assert reply == "The corrected result is 2."
    assert executions == ["1 / 0", "1 + 1"]
    assert selector.calls[1]["tool_results"][0]["error"]["code"] == "execution_error"


def test_needle2_native_response_is_adapted_without_executing_tools(monkeypatch):
    agents = []

    class FakeNeedle:
        def __init__(self, tools):
            self.tools = tools
            self.inputs = []
            self.reset_count = 0
            agents.append(self)

        def reset(self):
            self.reset_count += 1

        def complete(self, text):
            self.inputs.append(text)
            return {
                "type": "call",
                "confidence": 0.9,
                "function_calls": [{"name": "calculator", "arguments": {"expression": "2 + 2"}}],
            }

    monkeypatch.setitem(sys.modules, "needle", SimpleNamespace(Needle=FakeNeedle))
    selector = Needle2ToolSelector(minimum_confidence=0.5)
    schemas = [{"type": "function", "function": {
        "name": "calculator", "description": "Calculate.", "parameters": {"type": "object"},
    }}]

    first = selector.select([{"role": "user", "content": "Calculate 2 + 2"}], schemas)
    second = selector.select([], schemas, tool_results=[{"ok": True, "data": {"value": 4}}])

    assert '"name":"calculator"' in first
    assert '"name":"calculator"' in second
    active_agent = agents[-1]
    assert active_agent.tools[0]["name"] == "calculator"
    assert active_agent.reset_count == 1
    assert active_agent.inputs[1] == '{"ok":true,"data":{"value":4}}'


def test_needle2_rejects_calls_with_ungrounded_arguments(monkeypatch):
    class FakeNeedle:
        def __init__(self, tools):
            pass

        def reset(self):
            pass

        def complete(self, text):
            return {
                "type": "call", "success": True, "confidence": 0.9,
                "function_calls": [{"name": "weather", "arguments": {"place": "Current"}}],
                "validation": {"ungrounded": ["weather.place"]},
            }

    monkeypatch.setitem(sys.modules, "needle", SimpleNamespace(Needle=FakeNeedle))
    selector = Needle2ToolSelector()
    schemas = [{"type": "function", "function": {
        "name": "weather", "description": "Weather.", "parameters": {"type": "object"},
    }}]

    assert selector.select([{"role": "user", "content": "Hello"}], schemas) == ""


def test_unavailable_dedicated_selector_falls_back_to_main_model_protocol():
    class BrokenSelector(ToolSelector):
        identity = "needle2"

        def select(self, messages, schemas, tool_results=None):
            raise RuntimeError("runtime unavailable")

    main_model = QueueMainModel("qwen", [
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
        "The result is 4.",
    ])
    logs = []
    runner = HarnessRunner(
        main_model,
        FakeMemory(),
        tool_selector=BrokenSelector(),
        tool_manager=calculator_manager(lambda expression: {"value": 4}),
        logger=logs.append,
    )

    reply = runner.generate_tool_aware_reply(
        [{"role": "user", "content": "Calculate 2 + 2 using calculator."}],
        turn_number=1,
    )

    assert reply == "The result is 4."
    assert all(call.tools for call in main_model.calls)
    assert any(
        "selector=main_model_fallback outcome=structured_calls parsed_calls=1" in message
        for message in logs
    )
    assert any(
        "selector=main_model_fallback outcome=no_calls parsed_calls=0" in message
        for message in logs
    )


def test_dedicated_selector_allows_two_identical_successful_dice_calls():
    values = iter([3, 5])
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        "roll_die", "Roll one die.", lambda sides: {"value": next(values)},
        {
            "type": "object",
            "properties": {"sides": {"type": "integer", "minimum": 2}},
            "required": ["sides"],
            "additionalProperties": False,
        },
    ))
    calls = (
        '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
        '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
    )
    selector = QueueSelector([calls, ""])
    runner = HarnessRunner(
        QueueMainModel("smollm2", ["The rolls are 3 and 5."]),
        FakeMemory(),
        tool_selector=selector,
        tool_manager=ToolManager(registry),
        logger=lambda message: None,
    )

    reply = runner.generate_tool_aware_reply(
        [{"role": "user", "content": "Use roll_die twice with six sides."}],
        turn_number=1,
    )

    assert reply == "The rolls are 3 and 5."
    assert [entry.payload["data"]["value"] for entry in runner.last_tool_ledger] == [3, 5]


def test_dedicated_selector_can_use_prior_results_for_multi_tool_dependency():
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        "roll_die", "Roll one die.", lambda sides: {"value": 3},
        {"type": "object", "properties": {"sides": {"type": "integer"}}, "required": ["sides"]},
    ))
    registry.register(ToolDefinition(
        "random_number", "Generate a number.", lambda minimum, maximum: {"value": 2},
        {
            "type": "object",
            "properties": {"minimum": {"type": "integer"}, "maximum": {"type": "integer"}},
            "required": ["minimum", "maximum"],
        },
    ))
    registry.register(ToolDefinition(
        "calculator", "Calculate.", lambda expression: {"value": 5},
        {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    ))
    selector = QueueSelector([
        (
            '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
            '<tool_call>{"name":"random_number","arguments":{"minimum":1,"maximum":3}}</tool_call>'
        ),
        '<tool_call>{"name":"calculator","arguments":{"expression":"3 + 2"}}</tool_call>',
        "",
    ])
    runner = HarnessRunner(
        QueueMainModel("qwen", ["The total is 5."]),
        FakeMemory(),
        tool_selector=selector,
        tool_manager=ToolManager(registry),
        logger=lambda message: None,
    )

    reply = runner.generate_tool_aware_reply(
        [{"role": "user", "content": "Use roll_die and random_number, then calculator to add them."}],
        turn_number=1,
    )

    assert reply == "The total is 5."
    assert [entry.tool for entry in runner.last_tool_ledger] == ["roll_die", "random_number", "calculator"]
    first_results = selector.calls[1]["tool_results"]
    assert [result["data"]["value"] for result in first_results] == [3, 2]
